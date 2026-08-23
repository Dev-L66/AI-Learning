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

knowledge_base = {
    "age": "36",
    "net worth": "10 million"
}

def retrieve_info(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net worth"]
    else:
        return None
    

def ask_question(question):
    context = retrieve_info(question)
    system_prompt = f"""answer in one line only. Answer only based on this context. do not hallucinate. Context:{context}"""
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


question= "Mickey's net worth is? "

print(ask_question(question))