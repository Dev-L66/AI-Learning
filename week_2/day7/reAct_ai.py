import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from time import sleep
import re

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY not found.")

client = Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"

# tools
def get_product_price(product):
    if product == "iPhone 17":
        return 1000
    elif product == "iPhone 15":
        return 500
    else:
        return 0
    
def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc error!"
    
tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}

system_prompt = """
You are shopping assistant.

You have these tools:
get_product_price(product)
calculator(expression)

IMPORTANT:
Call tools exactly like these examples:
Action: get_product_price("iPhone 17")
Action: calculator("5000 - 1000")


Never write:
get_product_price(product= "iPhone 17")
calculator(expression = "5000 - 1000")


Follow these rules

1. Decide what you need to do next.
2. Call ONLY ONE tool at a time.
3. After writing an Action, STOP immediately.
4. Never guess or invent a tool result.
5. Wait until you receive an Observation.
6. Then decide your next action.
7. When the task is complete, give the Final Answer.

Format:

Thought: what you need to do
Action: tool_name(argument)

When finished:

Final Answer: your answer
"""
    

def run_agent(question):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
         {
            "role": "user",
            "content": question
        }
    ]

    for step in range(5):
        print("_______________")
        print("STEP", step + 1)
        print("_______________")

        response = client.chat.completions.create(model = model, messages = messages, temperature = 0)

        answer = response.choices[0].message.content

        print(answer)

        # Agent has finished
        if "Final Answer" in answer:
            break

        #  Find the Action
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')


        # Run the tool

        if tool_name in tools:
            tool = tools[tool_name]

            observation = tool(tool_input)
        else:
            observation = "Tool not found."

        
        print("Observation:", observation)


        #  Add LLM response to memory
        messages.append({
            "role":"assistant",
            "content": answer
        })


        #  Give tool result back to LLM
        messages.append({"role": "user",
                        "content": "Observation: " + str(observation)})
        # rate limit
        sleep(5)  
       




prompt = """
I have 5000 dollars.What is the price of an iPhone 17?
and how much money will I be left with?
"""

run_agent(prompt)
























# ReAct stands for Reason + Act. It is a prompting and agent framework that lets an LLM alternate between thinking about what to do next and using tools (such as search, databases, calculators, or APIs) until it can answer the user's question.

# The key idea is that the model doesn't try to answer everything from memory. Instead, it can gather information, perform actions, and then use the results to continue solving the problem.

# A typical ReAct loop looks like this:

# User Question
#       │
#       ▼
# Reason about next step
#       │
#       ▼
# Choose an action (tool)
#       │
#       ▼
# Receive observation (tool result)
#       │
#       ▼
# Reason again
#       │
#       ▼
# Repeat if needed
#       │
#       ▼
# Final answer

# Example
# Suppose the user asks:

# "What's the weather in Tokyo, and should I carry an umbrella?"

# A ReAct-style interaction would conceptually proceed like this:

# Thought:
# I need the current weather.

# Action:
# Call weather API for Tokyo.

# Observation:
# Light rain, 18°C.

# Thought:
# Since it's raining, an umbrella is recommended.

# Final Answer:
# It's currently 18°C with light rain in Tokyo. You should carry an umbrella.

# The "Thought" steps represent the model deciding what information it needs next. In production systems, these internal reasoning steps are generally not exposed to users. Instead, the system uses the reasoning internally and presents the actions taken and the final answer.

# Why ReAct is useful
# Without ReAct:

# Question
#    │
#    ▼
# LLM
#    │
#    ▼
# Answer

# The model relies only on what it already knows.

# With ReAct:

# Question
#    │
#    ▼
# LLM
#    │
#    ├── Search Web
#    ├── Query Database
#    ├── Run Calculator
#    ├── Read PDF
#    └── Call API
#          │
#          ▼
#     Observations
#          │
#          ▼
#    Better Answer

# This approach helps with:

# Up-to-date information (web search)
# Factual accuracy (retrieving data instead of guessing)
# Multi-step reasoning
# Tool use (calculators, code execution, APIs, databases)
# ReAct in code (conceptually)
# Many agent frameworks implement a loop similar to:

# while not finished:
#     thought = llm.decide_next_step()
#     action = choose_tool(thought)
#     observation = action.run()
#     conversation.append(observation)

# answer = llm.generate_final_answer()

# Libraries such as LangChain, LangGraph, and LlamaIndex build agent workflows around this general pattern, though each has its own implementation details.

# Example with a coding task
# User:

# "Summarize the README in this GitHub repository."

# A ReAct-style agent might:

# Recognize it needs the README.
# Fetch the repository.
# Read the README.
# Summarize the content.
# Return the summary.
# The important idea is that the model decides when external information is needed, uses a tool to obtain it, and then continues reasoning with the new information before producing the final response.



