import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

openai_api_client = OpenAI()


def generation(prompt_messages: list[ChatCompletionMessageParam]) -> str:
    response = openai_api_client.chat.completions.create(
        messages=prompt_messages,
        model='gpt-4o-mini',
        temperature=0,
    )

    return response.choices[0].message.content

def main():
    # Example prompt messages
    example_prompt = [
        {
            "role": "system",
            "content": "You are a battalion executive and operations officer specialized in leading and conducting military decision-making process."
        },
        {
            "role": "user",
            "content": "Construct a battalion strategy for regaining control of Kyiv."
        }
    ]
    
    try:
        response = generation(example_prompt)
        print("Assistant's response:")
        print(response)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__=='__main__':
    main()
