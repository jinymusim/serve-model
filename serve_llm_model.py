from flask import Flask, request, jsonify, Response
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import  AutoTokenizer, TextIteratorStreamer
from functools import wraps
import torch
import json
import os
import logging
from threading import Thread
from datetime import date

app = Flask(__name__)

model_dict = {
    "Llama-3.2-1B": "onnx-community/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B": "onnx-community/Llama-3.2-3B-Instruct"
}
models = {
    "Llama-3.2-1B": None,
    "Llama-3.2-3B": None
}

API_TOKEN = os.getenv("LOCAL_API_TOKEN", "metacentrum") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    filename="llm_server.log", 
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger("llm server")


def load_models():
    for model_id, model_name in model_dict.items():
        logger.info(f"Loading model {model_id} from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ORTModelForCausalLM.from_pretrained(model_name, subfolder="onnx/")
        model = model.to(DEVICE)
        models[model_id] = (model, tokenizer)


def generate_stream(prompt, max_tokens, temperature, model_id):
    model, tokenizer = models.get(model_id, (None, None))
    if model is None:
        return "Invalid model parameter"
    
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    logger.info(f"Input Tokens: {input_ids['input_ids'].shape[-1]}")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs =dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_p=0.99,
        top_k=1000,
        temperature=temperature,
        tokenizer=tokenizer,
        stop_strings=["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    tok_count = 0
    for token in streamer:
        tok_count += 1
        # Yield each token in OpenAI-compatible JSON format
        yield f"data: {json.dumps({'object': 'chat.completion.chunk','choices': [{'delta': {'content': token}, 'finish_reason': None} ]})}\n\n"

    logger.info(f"Output Tokens: {tok_count}")
    # Final message to signal end of streaming
    yield f"data: {json.dumps({'object': 'chat.completion.chunk','choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"


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

@app.route('/v1/completions/', methods=['POST'])
@requires_auth
def generate_completion():
    # Extract data from the request
    data = request.json
    prompt = "<|begin_of_text|>" + data.get('prompt', '')
    logger.info(f"Prompt: {prompt}")
    max_tokens = min(data.get('max_tokens', 100), 4096)
    logger.info(f"Request for {max_tokens} tokens")   
    model_id = data.get('model',None)
    logger.info(f"Model: {model_id}")
    temperature = data.get('temperature', 0.7)
    if model_id is None:
        return Response("Missing model parameter", 400)
    model, tokenizer = models.get(model_id, (None, None))
    if model is None:
        return Response("Invalid model parameter", 400)
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    logger.info(f"Input Tokens: {inputs['input_ids'].shape[-1]}")
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
    logger.info(f"Output Tokens: {len(outputs[0])}")
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
    max_tokens = min(data.get('max_tokens', 100), 4096)
    model_id = data.get('model', None)
    stream = data.get('stream', False)
    temperature = data.get('temperature', 0.7)
    if model_id is None:
        return Response("Missing model parameter", 400)
    model, tokenizer = models.get(model_id, (None, None))
    if model is None:
        return Response("Invalid model parameter", 400)
    
    # Prepare prompt from messages (chat history)
    prompt = "<|begin_of_text|>"
    for i, message in enumerate(messages):
        role = message.get('role')
        content = message.get('content')
        if i == 0 and role != "system":
            today = date.today().strftime("%B %d, %Y")
            prompt += f"<|start_header_id|>system<|end_header_id|>\n\nToday Date: {today}\nYou are a helpful assistant. You can answer questions about any topic.<|eot_id|>"
        if role and content:
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n {content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Request for {max_tokens} tokens")
    logger.info(f"Model: {model_id}")
    
    if stream:
        # Stream the completion in OpenAI-compatible format
        return Response(generate_stream(prompt, max_tokens, temperature ,model_id), content_type="text/event-stream")
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    logger.info(f"Input Tokens: {inputs['input_ids'].shape[-1]}")
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
    logger.info(f"Output Tokens: {len(outputs[0])}")
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
