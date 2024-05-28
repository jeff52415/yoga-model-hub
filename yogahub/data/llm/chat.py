import os

from openai import OpenAI

client = OpenAI(
    # Not necessary
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_chat_completion(prompt: str, model: str = "gpt-4o"):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    prompt = "Say Hello World!"
    response_content = generate_chat_completion(prompt)
    print(response_content)
