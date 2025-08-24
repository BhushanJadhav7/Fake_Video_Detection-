import os
from dotenv import load_dotenv

load_dotenv()

def get_env(name, default=None):
    return os.getenv(name, default)

def get_hf_api_key():
    key = os.getenv("HF_API_KEY")
    return key

def get_hf_model_id():
    return os.getenv("HF_MODEL_ID", "selimsef/xception")

def get_hf_api_url(model_id=None):
    model_id = model_id or get_hf_model_id()
    return f"https://api-inference.huggingface.co/models/{model_id}"
