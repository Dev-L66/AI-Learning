# # Use streaming when the consumer is a human reading the response.
# The user sees words appear as they're generated.
# This reduces the perceived wait time.
# It makes chat applications feel much more responsive.
# Don't use streaming by default when the consumer is another program expecting structured output (such as JSON).
# The program usually can't do anything useful until it has the complete JSON.
# Waiting for the full response is often simpler and less error-prone.
# Non-streaming code is generally easier to implement and debug.

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY not found.")
client = Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"

prompt = "What is Internet? Explain"

message = {
    "role": "user",
    "content": prompt
}
messages = [message]

stream_response = client.chat.completions.create(model=model, messages = messages, stream=True)

for chunk in stream_response:
    content = chunk.choices[0].delta.content
    if content:
     print(content, end="", flush=True)

