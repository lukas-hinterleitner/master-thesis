from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

load_dotenv()

__client = OpenAI()

__paraphrasing_system_prompt = """
You are a paraphrasing expert who is specialized in rewriting text (questions, statements, etc.) without altering the content. 
Keep in mind, that the meaning must not change after the paraphrasing. 
Just output the paraphrased text without any additional information.
"""

def paraphrase_input(input: str):
    response = __client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            ChatCompletionSystemMessageParam(content=__paraphrasing_system_prompt, role="system"),
            ChatCompletionUserMessageParam(content=input, role="user"),
        ],
        seed=42,
        temperature=1,
    )

    return response.choices[0].message.content