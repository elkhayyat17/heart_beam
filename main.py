import io
from fastapi import UploadFile, HTTPException, FastAPI
import librosa
import numpy as np
from model import predict
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(project_name="Dangerous Heartbeat Classification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World, Project name is : Dangerous Heartbeat Classification"}

@app.post("/heartChecking")
async def detect(voice: UploadFile):
    if voice.filename.split(".")[-1] not in ("wav", "mp3"):
        raise HTTPException(
            status_code=415, detail="Not a Voice"
        )

    audio_bytes = voice.file.read()

    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

    prediction = await predict(audio, sample_rate)

    if prediction == "murmur":
        f = open("HeartChecking/Murmur.txt", "r")
        return { "response": f.read() }
    elif prediction == "normal":
        f = open("HeartChecking/Normal.txt", "r")
        return { "response": f.read() }
    elif prediction == "artifact":
        f = open("HeartChecking/Artifact.txt", "r")
        return { "response": f.read() }
    elif prediction == "extrastole":
        f = open("HeartChecking/Extrasystole.txt", "r")
        return { "response": f.read() }
    elif prediction == "extrahls":
        f = open("HeartChecking/ExtraHeartSound.txt", "r")
        return { "response": f.read() }

    return {"response": prediction}

@app.post("/skinChecking")
async def detect(image: UploadFile):
    if image.filename.split(".")[-1] not in ('jpg', 'jpeg', 'png', 'gif', 'svg', 'bmp', 'webp', 'tiff'):
        raise HTTPException(
            status_code=415, detail="Not an image"
        )

    return {"response": "I'am working just fine."}

class ChatBotRequest(BaseModel):
    message: str

@app.post("/chatBot")
async def detect(request: ChatBotRequest):
    return { "response": "I'am working just fine and here is your message:\n"+ request.message }
