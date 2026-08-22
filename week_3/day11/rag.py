import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY not found.")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"



def ask_question(question):
    system_prompt = 'anwer in one line only'
    system_message ={
        "role": "system",
        "content": system_prompt
    }
    message={
    "role":"user",
    "content": question}


    messages = [message, system_message]
    response= client.chat.completions.create(model = model, messages=messages)
    answer = response.choices[0].message.content
    return answer


question= "Do you know padho with pratyush?"

print(ask_question(question))