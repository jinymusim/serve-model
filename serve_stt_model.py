from flask import Flask, request, jsonify, Response
from transformers import pipeline
from functools import wraps
import tempfile
import torch
import os
import logging

app = Flask(__name__)

model_dict = {
    "whisper-small": "openai/whisper-small",
}
models = {
    "whisper-small": None,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_TOKEN = os.getenv("LOCAL_API_TOKEN", "metacentrum") 

logging.basicConfig(
    filename="stt_server.log", 
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger("STT server")

def load_models():
    for model_id, model_name in model_dict.items():
        logger.info(f"Loading model {model_id} from {model_name}")
        model = pipeline("automatic-speech-recognition", model=model_name, device=DEVICE)
        models[model_id] = model

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        logger.info(f"Request from: {request.remote_addr}")
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.warning("Missing or invalid token")
            return Response("Missing or invalid token", 401)
        
        token = auth_header.split(" ")[1]
        if token != API_TOKEN:
            logger.warning("Unauthorized")
            return Response("Unauthorized", 403)
        
        logger.info("Authorized")
        return f(*args, **kwargs)
    return decorated

@app.route('/v1/audio/transcriptions', methods=['POST'])
@requires_auth
def transcribe_audio():
    # Check for audio file in the request
    if 'file' not in request.files:
        return Response("No audio file provided", 400)
    model_id = request.form.get("model", None)
    if model_id is None:
        return Response("Missing model parameter", 400)
    model = models.get(model_id, None)
    if model is None:
        return Response("Invalid model parameter", 400)
    
    # Save the file to a temporary location
    audio_file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name
        audio_file.save(audio_path)

    try:
        # Transcribe audio file
        transcription = model(audio_path)["text"]
        response = {
            "text": transcription,
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return Response(f"Internal Error", 500)

    finally:
        # Cleanup temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            

if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5002)