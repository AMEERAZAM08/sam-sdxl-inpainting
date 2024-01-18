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

def call_virtualmodel(input_image_url, input_mask_url, source_background_url, prompt, face_prompt):
    BATCH_SIZE=4
    headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "X-DashScope-Async": "enable",
        }
    data = {
            "model": "wanx-virtualmodel",
            "input":{
                "base_image_url": input_image_url,
                "mask_image_url": input_mask_url,
                "prompt": prompt,
                "face_prompt": face_prompt,
                "background_image_url": source_background_url,
            },
            "parameters": {
                "short_side_size": "512",
                "n": BATCH_SIZE
            }
        }
    url_create_task = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/virtualmodel/generation'
    res_ = requests.post(url_create_task, data=json.dumps(data), headers=headers)

    respose_code = res_.status_code
    if 200 == respose_code:
            res = json.loads(res_.content.decode())
            request_id = res['request_id']
            task_id = res['output']['task_id']
            logger.info(f"task_id: {task_id}: Create VirtualModel request success. Params: {data}")

            # 异步查询
            is_running = True
            while is_running:
                url_query = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
                res_ = requests.post(url_query, headers=headers)
                respose_code = res_.status_code
                if 200 == respose_code:
                    res = json.loads(res_.content.decode())
                    if "SUCCEEDED" == res['output']['task_status']:
                        logger.info(f"task_id: {task_id}: VirtualModel generation task query success.")
                        results = res['output']['results']
                        img_urls = []
                        for x in results:
                             if "url" in x:
                                  img_urls.append(x['url'])
                        logger.info(f"task_id: {task_id}: {res}")
                        break
                    elif "FAILED" != res['output']['task_status']:
                        logger.debug(f"task_id: {task_id}: query result...")
                        time.sleep(1)
                    else:
                        raise gr.Error('Fail to get results from VirtualModel task.')

                else:
                    logger.error(f'task_id: {task_id}: Fail to query task result: {res_.content}')
                    raise gr.Error("Fail to query task result.")

            logger.info(f"task_id: {task_id}: download generated images.")
            img_data = download_images(img_urls, len(img_urls)) if len(img_urls) > 0 else []
            logger.info(f"task_id: {task_id}: Generate done.")
            return img_data
    else:
            logger.error(f'Fail to create VirtualModel task: {res_.content}')
            raise gr.Error("Fail to create VirtualModel task.")