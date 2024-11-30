from flask import Flask, request, jsonify, Response
from sentence_transformers import SentenceTransformer
from functools import wraps
import torch
import os

app = Flask(__name__)

model_dict = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v1": "sentence-transformers/all-mpnet-base-v1"
}
models = {
    "all-MiniLM-L6-v2": None,
    "all-mpnet-base-v1": None
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    for model_id, model_name in model_dict.items():
        model = SentenceTransformer(model_name)
        model = model.to(DEVICE)
        models[model_id] = model

API_TOKEN = os.getenv("LOCAL_API_TOKEN", "metacentrum") 

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response("Missing or invalid token", 401)
        
        token = auth_header.split(" ")[1]
        if token != API_TOKEN:
            return Response("Unauthorized", 403)
        
        return f(*args, **kwargs)
    return decorated


@app.route('/v1/embeddings', methods=['POST'])
#@requires_auth
def generate_embedding():
    # Extract data from the request
    data = request.json
    text = data.get('input', '')
    model_id = data.get('model', None)
    if text == '':
        return Response("Missing input parameter", 400)
    if model_id is None:
        return Response("Missing model parameter", 400)
    model = models.get(model_id)
    if model is None:
        return Response("Invalid model parameter", 400)
    
    if not isinstance(text, list):
        text = [text]
    else:
        text = [str(t).strip() if str(t).strip() else 'NONE' for t in text]
    
    # Generate embedding
    embeddings = model.encode(text)
    response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding.tolist(),
                "index": i
            } for i, embedding in enumerate(embeddings)
        ]
    }
    
    # Initial response
    return jsonify(response)


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5001)
