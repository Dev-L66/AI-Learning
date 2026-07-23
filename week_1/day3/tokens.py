import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Key not found.")

client = Groq(api_key = my_api_key)

role = "user"
prompt1 = "Helllo, what's  up?"
prompt2 = "Explain time travel in detail"
prompt3 = "Write 1000 word essay on MAchine learning"
model = "openai/gpt-oss-20b"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:  
 message = {
     "role" : role,
    "content": prompt
}
 messages = [message]
 response = client.chat.completions.create(messages= messages, model = model, max_tokens = 100)
 usage = response.usage
 print(f"Prompt: {prompt} -->your tokens:{usage.prompt_tokens} completion_token:{usage.completion_tokens} total tokens:{usage.prompt_tokens + usage.completion_tokens }")
 print(f"Finish reason:{response.choices[0].finish_reason}")
#  print(response)
#  print(response.choices[0].message.content)


# A token is the basic unit of text that an AI model processes. 
# Tokens can be whole words, parts of words, punctuation, or other common text sequences. 
# The more tokens a model processes or generates, the more computation, time, and energy are required.

# Prompt tokens are the tokens in everything you send to the model.
# Completion tokens are the tokens the model generates in its response.

# finish_reason tells you why the model stopped generating tokens. It's included in API responses.
# Three typesof finish_reason stop, length and content_filter.
# stop:	The model finished naturally or encountered a stop sequence.
# length: The model stopped because it reached the maximum number of output tokens.
# content_filter: The response was stopped or filtered due to safety policies (available in some APIs/models).

