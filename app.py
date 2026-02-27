import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64
from io import BytesIO

# --- AI & DB CLIENTS ---
from groq import Groq
from supabase import create_client, Client
from elevenlabs.client import ElevenLabs
from diffusers import StableDiffusionPipeline
import torch

# --- CONFIGURATION ---
app = FastAPI(title="PromptPilot AI", version="GenZ-V5-Groq-Supabase")

# !!! KEYS (Use Environment Variables for Deployment) !!!
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "your_elevenlabs_key_here")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "your_supabase_project_url")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your_supabase_anon_key")

# --- CLIENT INITIALIZATION ---
# Groq Setup
groq_client = Groq(api_key=GROQ_API_KEY)
ACTIVE_GROQ_MODEL = "llama3-70b-8192" # Fast, highly capable model

# Supabase Setup
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase")
except Exception as e:
    print(f"⚠️ Supabase Setup Error: {e}")
    supabase = None

# ElevenLabs Setup
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY and ELEVENLABS_API_KEY != "your_elevenlabs_key_here" else None

# Vision Setup
image_pipe = None
def get_image_pipe():
    global image_pipe
    if image_pipe is None:
        print("Loading Vision Engine...")
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        image_pipe = image_pipe.to(device)
    return image_pipe

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    user_input: str
    user_profile: Optional[dict] = None
    use_voice: bool = False
    generate_image: bool = False

# --- CORE INTELLIGENCE ---
def interpret_intent(casual_input: str, profile: dict) -> str:
    name = profile.get('name', 'User')
    role = profile.get('role', 'Student')
    
    style_instruction = "Keep the tone helpful and friendly."
    if role == "Kid":
        style_instruction = "The user is a child. Rewrite the prompt to ask for an explanation that uses simple words, fun analogies, emoji, and short sentences. No complex jargon."
    elif role == "Student":
        style_instruction = "The user is a student. Rewrite the prompt to ask for a structured explanation with clear headings, bullet points, and an encouraging tone."
    elif role == "Professional":
        style_instruction = "The user is a professional. Rewrite the prompt to ask for a concise, executive-summary style answer with actionable steps and data-driven tone."

    system_instruction = (
        "You are the PromptPilot Kernel. Your goal is to rewrite the user's raw input into the PERFECT prompt for an AI LLM. "
        "Do not answer the question yourself. Just output the rewritten prompt. "
        f"Context: User is {name}, a {role}. {style_instruction}"
    )
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": casual_input}
            ],
            model=ACTIVE_GROQ_MODEL,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Interpretation Error: {e}")
        return casual_input

def execute_ai(professional_prompt: str, role: str) -> str:
    try:
        format_instruction = " Use clear Markdown formatting (bolding, bullet points) to make the text easy to read."
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": professional_prompt + format_instruction}
            ],
            model=ACTIVE_GROQ_MODEL,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Connection Error: {str(e)}"

# --- ENDPOINTS ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Handle Image Generation
    if request.generate_image:
        try:
            pipe = get_image_pipe()
            image = pipe(request.user_input).images[0]
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {"response": "Image generated.", "image_data": f"data:image/png;base64,{img_str}", "professional_prompt": request.user_input}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    # 2. Handle Groq Text Generation
    professional_prompt = interpret_intent(request.user_input, request.user_profile)
    role = request.user_profile.get('role', 'Student')
    response_text = execute_ai(professional_prompt, role)
    
    # 3. Handle ElevenLabs Voice Generation
    audio_data = None
    if request.use_voice and elevenlabs_client:
        try:
            audio = elevenlabs_client.generate(text=response_text[:400], voice="Rachel", model="eleven_turbo_v2")
            audio_data = base64.b64encode(b"".join(audio)).decode()
        except: pass

    # 4. Log to Supabase
    if supabase:
        try:
            supabase.table("chat_logs").insert({
                "user_name": request.user_profile.get('name', 'Unknown'),
                "user_role": role,
                "raw_input": request.user_input,
                "rewritten_prompt": professional_prompt,
                "ai_response": response_text
            }).execute()
        except Exception as e:
            print(f"Failed to log to Supabase: {e}")

    return {
        "response": response_text,
        "professional_prompt": professional_prompt,
        "audio_data": audio_data
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    return {"filename": file.filename, "full_content": "Document processing placeholder"}

# --- GEN Z UI (Unchanged) ---
# Paste your entire HTML string exactly as it was here.
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>...</head><body>...</body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_ui(): return html_content

if __name__ == "__main__":
    # Uses the PORT environment variable if available (required for cloud deployment)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
    