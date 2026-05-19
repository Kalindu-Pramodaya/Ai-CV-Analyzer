import os
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
def home():
    return {"message": "Premium AI CV Matcher is Online."}

@app.post("/analyze")
async def analyze_cv(
    cv_file: UploadFile = File(...), 
    job_description: str = Form(...)
):
    try:
        pdf_content = await cv_file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        cv_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                cv_text += text + "\n"
        
        prompt = f"""
        You are an expert recruitment consultant and ATS specialist in the United Kingdom. 
        Analyze the candidate's CV against the provided Job Description.
        
        [CANDIDATE CV]
        {cv_text}
        
        [JOB DESCRIPTION]
        {job_description}
        
        STRICT FORMATTING RULES:
        1. Start the very first line with "Match Percentage: XX%" (e.g., Match Percentage: 85%).
        2. Use '###' before every new section heading.
        3. You MUST include these three specific sections:
           ### ATS Keyword Gaps
           ### UK Market Strategy
           ### Recommended Profile Edits
        4. Be blunt, professional, and provide actionable advice for the UK tech market.
        """
        
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite', 
            contents=prompt,
        )
        
        return {
            "status": "success",
            "analysis": response.text
        }
        
    except Exception as e:
        print(f"Server Error: {e}") 
        return {"status": "error", "message": str(e)}
