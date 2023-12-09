import io
from fastapi import APIRouter, UploadFile, HTTPException, FastAPI
import librosa
import numpy as np
from model import predict
from fastapi.middleware.cors import CORSMiddleware

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
