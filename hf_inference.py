import json
import requests

def infer_image_bytes(image_bytes, api_key, model_id, timeout=60):
    """Send a single image (bytes) to the HF Inference API model and return JSON output.
    Raises RuntimeError for 503 (model loading) or non-JSON responses.
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(url, headers=headers, data=image_bytes, timeout=timeout)
    if resp.status_code == 503:
        try:
            info = resp.json()
        except Exception:
            info = {"error": resp.text}
        raise RuntimeError(f"Model {model_id} is loading or unavailable: {info}")
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"Unexpected non-JSON response: {resp.text[:200]}")

def parse_fake_probability(api_output):
    """Parse a 'fake/manipulated probability' from common HF model outputs.
    Returns a float in [0,1]. Raises ValueError if format unrecognized.
    """
    # Common case: list of {label, score}
    if isinstance(api_output, list):
        # Prefer explicit 'fake-like' labels
        for item in api_output:
            if isinstance(item, dict):
                label = str(item.get('label','')).lower()
                if label in ('fake','manipulated','deepfake'):
                    return float(item['score'])
        # If both real/fake exist, prefer fake score (else 1 - real)
        labels = [str(x.get('label','')).lower() for x in api_output if isinstance(x, dict)]
        if 'fake' in labels and 'real' in labels:
            for item in api_output:
                if str(item.get('label','')).lower() == 'fake':
                    return float(item['score'])
            for item in api_output:
                if str(item.get('label','')).lower() == 'real':
                    return 1.0 - float(item['score'])
        # Fallback: take max score as probability of manipulation
        scores = [float(x.get('score')) for x in api_output if isinstance(x, dict) and 'score' in x]
        if scores:
            return max(scores)

    # Dict case: {label, score}
    if isinstance(api_output, dict) and 'score' in api_output:
        label = str(api_output.get('label','')).lower()
        s = float(api_output['score'])
        if label in ('real','genuine','authentic'):
            return 1.0 - s
        return s

    raise ValueError(f"Unrecognized model output: {api_output}")
