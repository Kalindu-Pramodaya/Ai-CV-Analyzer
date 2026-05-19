import os
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

# 1. Setup and Security
load_dotenv()
app = FastAPI()

# Allow your frontend to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configure AI
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = os.getenv("AI_MODEL_NAME", "gemini-3.1-flash-lite")

# 3. Serve Frontend Files
# This tells FastAPI to serve all files inside your 'frontend' folder
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def read_index():
    # This sends your index.html whenever someone visits your main URL
    return FileResponse('frontend/index.html')

# 4. The AI Analysis Logic
@app.post("/analyze")
async def analyze_cv(file: UploadFile = File(...), job_description: str = Form(...)):
    try:
        # Read the PDF content
        pdf_content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        cv_text = ""
        for page in pdf_reader.pages:
            cv_text += page.extract_text()

        # Build the prompt for Gemini
        prompt = f"""
        You are an expert UK recruiter. Compare the following CV text with the Job Description.
        Provide a Match Percentage (0-100%) and a brief summary of missing skills or improvements.
        
        Job Description: {job_description}
        CV Content: {cv_text}
        """

        # Get response from AI
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )

        return {"analysis": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
