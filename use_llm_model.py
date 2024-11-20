from openai import OpenAI

# Override the OpenAI API's base URL and API key
api_base = "http://localhost:5000/v1"
api_embed = "http://localhost:5001/v1"
api_stt = "http://localhost:5002/v1"
api_key = "metacentrum"  # Replace with your API token

client = OpenAI(api_key=api_key, base_url=api_base)
embed_client = OpenAI(api_key=api_key, base_url=api_embed)
stt_client = OpenAI(api_key=api_key, base_url=api_stt)


# Example of using the completions endpoint
def get_completion(prompt):
    response = client.completions.create(
        model="Llama-3.2-1B",  # This model name is for compatibility; replace as needed
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text

# Example of using the chat completions endpoint
def get_chat_completion(messages):
    response = client.chat.completions.create(
        model="Llama-3.2-3B",  # This model name is for compatibility; replace as needed
        messages=messages,
        max_tokens=1024,
        stream=True
    )
    print("Streaming response:", end=" ")
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)

def get_embedding(text):
    response = embed_client.embeddings.create(
        model="all-mpnet-base-v1",
        input=text
    )
    return response.data[0].embedding

def transcribe_audio(audio_file):
    response = stt_client.audio.transcriptions.create(
        model="whisper-small",
        file=audio_file
    )
    return response.text

# Test with a prompt (completions endpoint)
prompt = "What is the capital of France?"
#print("Completion:", get_completion(prompt))

# Test with a chat conversation (chat completions endpoint)
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "Can you tell me more about it?"}
]
#get_chat_completion(messages)


# Test with an embedding request
text = "This is a test sentence."
#print("Embedding:", get_embedding(text))


# Test with an audio transcription request
audio_file = "CantinaBand3.wav"
file_obj = open(audio_file, "rb")
print("Transcription:", transcribe_audio(file_obj))
