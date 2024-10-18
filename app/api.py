import base64
import io
import os
from typing import List

import cv2
import numpy as np
import requests
import tensorflow.keras.backend as K
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import RedirectResponse
from tensorflow.keras.models import load_model

from app.models import RecordBaseResponse, RecordsBaseResponse
from . import config

prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").rstrip("/")

app = FastAPI(
    title="OCR",
    version="1.0",
    description="OCR",
    openapi_prefix=prefix,
)

# model_names = ['garena', 'zalo', 'vietcombank', 'csgt', 'gplx', 'payoo_web']
# model_names = ['dichvucong', 'xuatnhapcanh', 'vietcombank2', 'vcb_nganluong', 'evisa']
model_names = ['evisa']

model_inferences = {}
configs = {}


@app.on_event('startup')
async def load_config():
    global model_inferences, configs
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    for name in model_names:
        model_inference = load_model(os.path.join(dir_path, f'{name}.h5'))
        model_inferences[name] = model_inference

        config.load_config(os.path.join(dir_path, f'{name}.json'))
        configs[name] = {
            "height": config.DatasetConfig.height,
            "depth": config.DatasetConfig.depth,
            "max_len": config.DatasetConfig.max_len,
            "charset": config.DatasetConfig.charset,
            "normalize": config.DatasetConfig.normalize
        }


def predict(model_name, img):
    h, w = img.shape[:2]
    new_h = configs[model_name]['height']
    w = int(w * new_h / h)
    img = cv2.resize(img, (w, new_h))
    if configs[model_name]['depth'] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    # if configs[model_name].get('normalize'):
    #     img = img - 0.5
    x = np.expand_dims(img, axis=0)
    y_pred = model_inferences[model_name].predict(x)
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:,
          :configs[model_name]['max_len']]
    out = ''.join([configs[model_name]['charset'][x] if x >= 0 else '' for x in out[0]])
    return out


async def file_to_image(file):
    """Parser image in bytes"""
    npimg = np.frombuffer(await file.read(), np.int8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img


async def base64_to_image(img_str):
    imgdata = base64.b64decode(str(img_str))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img


async def url_to_image(url):
    res = requests.get(url)
    npimg = np.frombuffer(res.content, np.int8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img


async def preprocess_image_evisa(image_str):
    image_str = image_str.replace('data:image/png;base64,', '')
    img_data = base64.b64decode(image_str)
    img = Image.open(io.BytesIO(img_data))
    # img = img.resize((120, 50))
    arr = np.array(img)

    arr[arr == 0] = 255
    arr = np.expand_dims(arr, axis=-1)
    arr = np.concatenate((arr, arr, arr), axis=-1)
    return arr


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"{prefix}/docs")


@app.post("/api/v1/captcha/{model_name}", response_model=RecordBaseResponse, tags=['captcha'])
async def ocr_captcha_route(model_name: str, files: UploadFile = File(...)):
    if model_name not in model_names:
        return HTTPException(status_code=400, detail="Model not implemented")
    image = await file_to_image(files)
    result = predict(model_name, image)
    return RecordBaseResponse(result=result)


class ImageStr(BaseModel):
    image: str


@app.post("/api/v1/captcha/{model_name}/base64", response_model=RecordBaseResponse, tags=['captcha'])
async def ocr_captcha_route(model_name: str, image: ImageStr):
    if model_name not in model_names:
        return HTTPException(status_code=400, detail="Model not implemented")
    if model_name == 'evisa':
        image = await preprocess_image_evisa(image.image)
    else:
        image = await base64_to_image(image.image)
    result = predict(model_name, image)
    return RecordBaseResponse(result=result)


@app.post("/api/v1/captcha/{model_name}/url", response_model=RecordBaseResponse, tags=['captcha'])
async def ocr_captcha_route(model_name: str, image: ImageStr):
    if model_name not in model_names:
        return HTTPException(status_code=400, detail="Model not implemented")
    image = await url_to_image(image.image)
    result = predict(model_name, image)
    return RecordBaseResponse(result=result)


@app.post("/api/v1/captcha/{model_name}/multi", response_model=RecordsBaseResponse, tags=['captcha'])
async def ocr_captcha_multi_route(model_name: str, files: List[UploadFile] = File(...)):
    if model_name not in model_names:
        return HTTPException(status_code=400, detail="Model not implemented")
    results = []
    for file in files:
        image = await file_to_image(file)
        result = predict(model_name, image)
        results.append(result)
    return RecordsBaseResponse(result=results)


@app.get('/health-check')
async def health_check(request: Request) -> str:
    return "Ok"


if __name__ == '__main__':
    uvicorn.run('api:app', port=15004, debug=True)
