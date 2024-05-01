import os
from typing import List, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from yogahub.models import YogaClassifier

app = FastAPI()
model = YogaClassifier()


class PredictionRequest(BaseModel):
    convert_to_chinese: bool = True


class PredictionResponse(BaseModel):
    Target: str
    PotentialCandidate: Union[List[str], str]
    Gesture: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...), convert_to_chinese: bool = True):
    try:
        image_data = await image.read()
        # Convert the bytes to a NumPy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Convert to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        if height < 64 or width < 64:
            raise ValueError("Image size is too small")

        output = model.predict(images=img_rgb, convert_to_chinese=convert_to_chinese)
        return PredictionResponse(**output)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def run():
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", False)
    uvicorn.run("main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
