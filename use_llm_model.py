from openai import OpenAI

# Override the OpenAI API's base URL and API key
api_base = "http://localhost:5000/v1"
api_embed = "http://localhost:5001/v1"
api_key = "metacentrum"  # Replace with your API token

client = OpenAI(api_key=api_key, base_url=api_base)
embed_client = OpenAI(api_key=api_key, base_url=api_embed)


# Example of using the completions endpoint
def get_completion(prompt):
    response = client.completions.create(
        model="small_llm",  # This model name is for compatibility; replace as needed
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text

# Example of using the chat completions endpoint
def get_chat_completion(messages):
    response = client.chat.completions.create(
        model="bigger_llm",  # This model name is for compatibility; replace as needed
        messages=messages,
        max_tokens=1024,
        stream=True
    )
    print("Streaming response:", end=" ")
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)

def get_embedding(text):
    response = embed_client.embeddings.create(
        model="bigger_vectors",
        input=text
    )
    return response.data[0].embedding

# Test with a prompt (completions endpoint)
prompt = "What is the capital of France?"
#print("Completion:", get_completion(prompt))

# Test with a chat conversation (chat completions endpoint)
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "Can you tell me more about it?"}
]
get_chat_completion(messages)


# Test with an embedding request
text = "This is a test sentence."
#print("Embedding:", get_embedding(text))
