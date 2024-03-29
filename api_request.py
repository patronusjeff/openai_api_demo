from openai import OpenAI

base_url = "http://0.0.0.0:9191/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)


def simple_chat(use_stream=False):
    messages = [
        {
            "role": "user",
            "content": "Who is the author of Harry Potter?"
        }
    ]

    response = client.chat.completions.create(
        model="neural-chat-7b-v3-3",
        messages=messages,
        stream=use_stream,
        max_tokens=1024,
        presence_penalty=1.1,
    )

    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content)
        else:
            content = response.choices[0].message.content
            print(content)
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    simple_chat(use_stream=True)
    simple_chat(use_stream=False)
