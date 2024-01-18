import os
import numpy
from PIL import Image
import requests
import urllib.request
from http import HTTPStatus
from datetime import datetime
import json
from .log import logger
import time
import gradio as gr
from .util import download_images

API_KEY = os.getenv("API_KEY_VIRTUALMODEL")

def call_person_detect(input_image_url):
    headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "X-DashScope-DataInspection": "enable",
        }
    data = {
            "model": "body-detection",
            "input":{
                "image_url": input_image_url,
            },
            "parameters": {
                "score_threshold": 0.6,
            }
        }
    url_create_task = 'https://dashscope.aliyuncs.com/api/v1/services/vision/bodydetection/detect'
    res_ = requests.post(url_create_task, data=json.dumps(data), headers=headers)


    res = json.loads(res_.content.decode())
    request_id = res['request_id']
    results = res['output']['results']
    return results