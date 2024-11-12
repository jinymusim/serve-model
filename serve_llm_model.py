from flask import Flask, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from functools import wraps
import torch
import json
import os
import time
from threading import Thread

app = Flask(__name__)

model_dict = {
    "small_llm": "unsloth/Llama-3.2-1B-Instruct",
    "bigger_llm": "unsloth/Llama-3.2-3B-Instruct"
}
models = {
    "small_llm": None,
    "bigger_llm": None
}



BATCH_SIZE = 4
BATCH_INTERVAL = 1.0
API_TOKEN = os.getenv("LOCAL_API_TOKEN", "metacentrum") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    for model_id, model_name in model_dict.items():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        model = model.to(DEVICE)
        models[model_id] = (model, tokenizer)


def generate_stream(prompt, max_tokens,temprature, model_id):
    model, tokenizer = models.get(model_id, (None, None))
    if model is None:
        return "Invalid model parameter"
    
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs =dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.99,
        top_k=1000,
        temperature=temprature,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    for token in streamer:
        # Yield each token in OpenAI-compatible JSON format
        yield f"data: {json.dumps({'object': 'chat.completion.chunk','choices': [{'delta': {'content': token}, 'finish_reason': None} ]})}\n\n"

    # Final message to signal end of streaming
    yield f"data: {json.dumps({'object': 'chat.completion.chunk','choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"


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

@app.route('/v1/completions/', methods=['POST'])
@requires_auth
def generate_completion():
    # Extract data from the request
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    model_id = data.get('model',None)
    temprature = data.get('temperature', 0.7)
    if model_id is None:
        return Response("Missing model parameter", 400)
    model, tokenizer = models.get(model_id, (None, None))
    if model is None:
        return Response("Invalid model parameter", 400)
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_tokens, do_sample=True, temprature=temprature)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Format response similar to OpenAI's API
    response = {
        "choices": [
            {
                "text": completion,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ]
    }
    return jsonify(response)

@app.route('/v1/chat/completions', methods=['POST'])
@requires_auth
def generate_chat_completion():
    # Extract chat history and max_tokens from the request
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 50)
    model_id = data.get('model', None)
    stream = data.get('stream', False)
    temprature = data.get('temperature', 0.7)
    if model_id is None:
        return Response("Missing model parameter", 400)
    model, tokenizer = models.get(model_id, (None, None))
    if model is None:
        return Response("Invalid model parameter", 400)
    
    # Prepare prompt from messages (chat history)
    prompt = ""
    for message in messages:
        role = message.get('role')
        content = message.get('content')
        if role and content:
            prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant: "
    
    if stream:
        # Stream the completion in OpenAI-compatible format
        return Response(generate_stream(prompt, max_tokens, temprature ,model_id), content_type="text/event-stream")
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_tokens, do_sample=True, temprature=temprature)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Format response similar to OpenAI's API
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": completion
                },
                "index": 0,
                "finish_reason": "length"
            }
        ]
    }
    return jsonify(response)

if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000)
