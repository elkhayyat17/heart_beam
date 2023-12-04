import io
from fastapi import APIRouter, UploadFile, HTTPException, FastAPI
import librosa
import numpy as np
from model import predict

app = FastAPI(project_name="Dangerous Heartbeat Classification")

@app.get("/")
async def read_root():
    return {"Hello": "World, Project name is : Dangerous Heartbeat Classification"}

@app.post("/prediction")
async def detect(voice: UploadFile):
    if voice.filename.split(".")[-1] not in ("wav", "mp3"):
        raise HTTPException(
            status_code=415, detail="Not a Voice"
        )

    audio_bytes = voice.file.read()

    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

    prediction = await predict(audio, sample_rate)

    return {"prediction": prediction}
