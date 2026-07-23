import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY not found")

client = Groq(api_key = my_api_key)

model = "llama-3.3-70b-versatile"
role = "user"

class Ticket(BaseModel):
    name:str
    email:str
    issue:str
schema = Ticket.model_json_schema()

response_format = {
    "type":"json_object"
}
system_prompt = f"""
Extract personal info from the ticket strictly based on ths schema and give the output in json format. {schema}
"""

text = "Hello, my name is Ali. My friend's name is An. I have a MAC that isn't working. My email is xyz@gmail.com. My contact number is 6767676"

prompt = f"""
This is a customer ticket. Please extract personal info from this {text}
"""
message_system = {
    "role": "system",
    "content": system_prompt
}

message = {
    "role": role,
    "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model = model, messages = messages, response_format = response_format)

answer = response.choices[0].message.content

import json
raw_json = answer
data_file = json.loads(raw_json)
ticket =Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)