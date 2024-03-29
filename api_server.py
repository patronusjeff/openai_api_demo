# Usage:
#   python api_server.py

import time
import asyncio
from queue import Queue
from threading import Thread
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from loguru import logger
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.generation.logits_process import LogitsProcessor

from sse_starlette.sse import EventSourceResponse

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

streamer_queue = Queue()


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    tool_calls: Optional[list] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    tool_calls: Optional[list] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class LLMStreamer(TextStreamer):
    def __init__(self, queue, tokenizer, skip_prompt, **decode_kwargs) -> None:
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._queue = queue
        self.stop_signal = None
        self.timeout = 1

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text != "":
            self._queue.put(text)
        if stream_end:
            self._queue.put(self.stop_signal)


def process_messages(messages, tools=None):
    _messages = messages
    messages = []
    if tools:
        """
            generate tool system content
            messages.append(
                {
                    "role": "system",
                    "content": content
                }
            )
        """
        pass

    for m in _messages:
        role, content, tool_calls = m.role, m.content, m.tool_calls
        if role == "function":
            messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == "assistant" and tool_calls is not None:
            pass
        else:
            messages.append({"role": role, "content": content})
    return messages


def build_chat_input(query, history, role):
    if history is None:
        history = []
    prompt = ""

    for item in history:
        content = item["content"]
        if item["role"] == "system":
            prompt += "### System:\n"
            prompt += content
            prompt += "\n"
        if item["role"] == "user":
            prompt += "### User:\n"
            prompt += content
            prompt += "\n"
        if item["role"] == "assistant":
            prompt += "### Assistant:\n"
            prompt += content
            prompt += "\n"
        if item["role"] == "observation":
            # prompt += "### Function:\n"
            # prompt += content
            # prompt += "\n"
            pass

    if role == "observation":
        pass
    else:
        prompt += "### User:\n"
    prompt += query
    prompt += "\n### Assistant:\n"
    return prompt


@torch.inference_mode()
def start_generation(model, tokenizer, params):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    tools = params["tools"]

    messages = process_messages(messages, tools=tools)
    query, role = messages[-1]["content"], messages[-1]["role"]
    prompt = build_chat_input(query=query, history=messages[:-1], role=role)
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)

    # eos_token_id = [
    #     tokenizer.eos_token_id
    # ]

    streamer = LLMStreamer(streamer_queue, tokenizer, True, skip_special_tokens=True)

    gen_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 1e-5 else False,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        logits_processor=[InvalidScoreLogitsProcessor()],
        eos_token_id=2,
        pad_token_id=2,
        temperature=temperature,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()


async def generator_with_stream(model, tokenizer, params):
    start_generation(model, tokenizer, params)

    model_id = params.get("model_id", "neural-chat-7b-v3-3")
    while True:
        value = streamer_queue.get()
        if value is None:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason="stop"
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id="",
                choices=[choice_data],
                created=int(time.time()),
                object="chat.completion.chunk"
            )
            yield "{}".format(chunk.json(exclude_unset=True))
            yield "[DONE]"
            break

        message = DeltaMessage(
            content=value,
            role="assistant",
            tool_calls=None,
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=message,
            finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id="",
            choices=[choice_data],
            created=int(time.time()),
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.json(exclude_unset=True))

        streamer_queue.task_done()
        await asyncio.sleep(0.1)


@torch.inference_mode()
def generator_without_stream(model, tokenizer, params):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    tools = params["tools"]
    messages = process_messages(messages, tools=tools)

    query, role = messages[-1]["content"], messages[-1]["role"]
    prompt = build_chat_input(query=query, history=messages[:-1], role=role)
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    # eos_token_id = [
    #     tokenizer.eos_token_id
    # ]

    gen_kwargs = dict(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 1e-5 else False,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        logits_processor=[InvalidScoreLogitsProcessor()],
        eos_token_id=2,
        pad_token_id=2,
        temperature=temperature,
    )

    output = model.generate(**gen_kwargs).cpu()
    total_len = len(output[0])
    if echo:
        text = tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        text = tokenizer.decode(output[0][input_echo_len:], skip_special_tokens=True)
    print(text)
    return {
        "text": text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len
        }
    }


def parse_response(gen_result):
    pass


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="neural-chat-7b-v3-3")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        model_id=request.model,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        if request.tools:
            raise HTTPException(
                status_code=400,
                detail=
                "Invalid request: Function calling is not yet implemented for stream mode.",
            )
        return EventSourceResponse(generator_with_stream(model, tokenizer, gen_params), media_type="text/event-stream")

    else:
        gen_result = generator_without_stream(model, tokenizer, gen_params)
        tool_calls, finish_reason = None, "stop"
        text = gen_result["text"]
        if request.tools:
            # text, tool_calls = parse_response(gen_result)
            pass
        if tool_calls is not None and len(tool_calls) > 0:
            finish_reason = "tool_calls"

        message = ChatMessage(
            role="assistant",
            content=text,
            tool_calls=tool_calls
        )

        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )

        usage = UsageInfo(**gen_result["usage"])

        return ChatCompletionResponse(
            model=request.model,
            id="",
            choices=[choice_data],
            object="chat.completion",
            usage=usage
        )


if __name__ == "__main__":
    model_name = "Intel/neural-chat-7b-v3-3"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.eval()
    uvicorn.run(app, host="0.0.0.0", port=9191, workers=1)
