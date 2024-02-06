import io
from fastapi import UploadFile, HTTPException, FastAPI
import librosa
from model import predict ,Diagnose 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from PIL import Image


app = FastAPI(project_name="Dangerous Heartbeat Classification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/heartChecking")
async def detect(voice: UploadFile):
    if voice.filename.split(".")[-1] not in ("wav", "mp3"):
        raise HTTPException(
            status_code=415, detail="Not a Voice"
        )

    audio_bytes = voice.file.read()
    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
    prediction = await predict(audio, sample_rate)

    response = json.load(open("heartChecking/response.json"))

    match prediction:
        case "murmur":
            return response["murmur"]
        case "normal":
            return response["normal"]
        case "artifact":
            return response["artifact"]
        case "extrastole": 
            return response["extrasystole"]
        case "extrahls":
            return response["extraHeartSound"]


@app.post("/skinChecking")
async def detect(image: UploadFile):
    if image.filename.split(".")[-1] not in ('jpg', 'jpeg', 'png', 'gif', 'svg', 'bmp', 'webp', 'tiff'):
        raise HTTPException(
            status_code=415, detail="Not an image"
        )
    # Load the image

    prediction = await  Diagnose(image.file.read())

    return prediction 

class ChatBotRequest(BaseModel):
    message: str

@app.post("/chatBot")
async def detect(request: ChatBotRequest):
    return { "response": "I'am working just fine and here is your message:\n"+ request.message }
