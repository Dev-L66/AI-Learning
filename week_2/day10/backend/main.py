from fastapi import FastAPI, Request
from pathlib import Path
from pypdf import PdfReader
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import json
import os
 
load_dotenv()


my_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key = my_api_key)

model = "openai/gpt-oss-120b"

class ChatRequest(BaseModel):
   question: str

class Experience(BaseModel):
   company: str | None = None
   role: str | None = None
   duration: str | None = None
   description: str | None = None
   skills_used: list[str] = []

class Resume(BaseModel):
   name:str | None = None
   email:str | None = None
   phone: str | None = None
   total_experience_years: float | None = None
   skills: list[str] = []
   experiences: list[Experience]= []
   education: list[str] = []
   projects: list[str] = []
   certifications: list[str] = []

resume_schema = Resume.model_json_schema()

# parsing resume

def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """

    message_system = {
       "role": "system",
       "content": system_prompt
     }
    
    message_user = {
       "role": "user",
       "content" : user_prompt
    }

    messages = [message_system, message_user]
    response_format = {
       "type": "json_object"
    }
    response = client.chat.completions.create(model= model, messages= messages, response_format= response_format)
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume


def ask_candidate(question:str, resume:Resume):
    system_prompt = f"""
You are an assistant AI representing job candidate.
Review is everything you know about the candidate.
{resume.model_dump_json(indent = 2)}

Rules:
1. Answer only using this information.
2. Never hallucinate.
3. If information is unavailable say "I don't have enough information to answe that"
4. Be professional.
5. Answer as if you are interviweing the candidate.


"""
    messages = [
{
   "role": "system",
   "content": system_prompt},
   {"role": "user",
   "content":question }]

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


app =FastAPI()


# pdf extraction
def read_pdf(file_path: Path):
 reader = PdfReader(file_path)
 text = ""
 for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
       text += page_text + "\n"
 return text



@app.get("/")
def home():
    resume_text = read_pdf(Path("CV - SB.pdf"))
    resume = parse_resume(resume_text)
    print(resume.model_dump_json(indent=2))
    return {
        "message":"homepage"
    }



@app.post("/chat")
def chat(request:ChatRequest):
    resume_text = read_pdf(Path("CV - SB.pdf"))
    resume = parse_resume(resume_text)
    answer = ask_candidate(request.question, resume)
    return {"answer": answer}
        