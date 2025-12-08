import json
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image
import onnxruntime as ort


MODEL_PATH = "hair_classifier_v1.onnx"

# ImageNet normalization used in HW8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = (200, 200)  # width, height


def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img: Image.Image) -> np.ndarray:
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to 200x200
    img = img.resize(TARGET_SIZE, Image.NEAREST)

    # Convert to numpy, shape (H, W, C), values [0, 255]
    img_np = np.array(img).astype(np.float32) / 255.0  # scale to [0, 1]

    # Normalize with ImageNet mean/std
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    # Change to (C, H, W)
    img_np = np.transpose(img_np, (2, 0, 1))

    # Add batch dimension: (1, C, H, W)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np.astype(np.float32)


# -----------------------
# Load ONNX model once (cold start)
# -----------------------
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name



def lambda_handler(event, context):
    """
    Expects a JSON body with:
    {
        "url": "https://....jpeg"
    }

    Returns:
    {
        "prediction": float,   # raw model output (can be negative / >1)
        "url": "...",
    }
    """
    try:
        # API Gateway proxy integration usually gives body as string
        body = event.get("body", event)

        if isinstance(body, str):
            body = json.loads(body)

        url = body.get("url")
        if not url:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'url' in request body"})
            }

        img = download_image(url)
        x = prepare_image(img)

        # Run ONNX model
        pred = session.run([output_name], {input_name: x})[0][0][0]
        pred_float = float(pred)

        result = {
            "url": url,
            "prediction": pred_float
        }

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


# For local testing (python lambda_function.py)
if __name__ == "__main__":
    test_event = {
        "body": json.dumps({
            "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
        })
    }
    resp = lambda_handler(test_event, None)
    print(resp)
