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
            "content": "I am an infantry battalion commander, and have been asked to secure a airstrip on Sicily Drop Zone, with a time on target of 0400 on 1 March, 2025.  My battalion, 2-508th PIR, will be augmented with an SAPPER engineer platoon and M777 artillary battery.  Draft a warning order, consistent with FM 5.0, in the 5 paragraph oporder format."
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
