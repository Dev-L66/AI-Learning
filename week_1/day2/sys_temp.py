import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key = my_api_key)

model = "llama-3.3-70b-versatile"
role = "user"
prompt = "shut up and dance with me"

#System
message_system = {
    "role" :"system",
    "content" : "You are my interviewer."
}

#message has role and content
message = {
    "role" : role,
    "content": prompt
}
messages = [message_system, message]

#Temperature by default is 0 meaning safe, range is [0,2]
response = client.chat.completions.create(model=model, messages=messages, temperature=2)
print(response.choices[0].message.content)

# System Role
# The system role is a special instruction that tells the model how it should behave throughout the conversation.
# # | Role          | Purpose                                         |
# | ------------- | ----------------------------------------------- |
# | **system**    | Sets the assistant's behavior and instructions. |
# | **user**      | Contains the user's requests or questions.      |
# | **assistant** | Contains the model's previous responses.        |

# Temperature controls how random or creative the model's responses are.

# Low temperature → More focused and consistent.
# High temperature → More varied and creative.
# 0.0 to 2.0 range of temperature.
# Temperature = 0, The model is very deterministic.
# Temperature = 1.5, Highly creative and less predictable.


# System role answers "Who should the AI be?"
# Temperature answers "How predictable or creative should the AI's responses be?"