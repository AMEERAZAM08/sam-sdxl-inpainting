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

def call_bg_genration(base_image, ref_img, prompt,ref_prompt_weight=0.5):
    API_KEY = os.getenv("API_KEY_BG_GENERATION")
    BATCH_SIZE=4
    headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "X-DashScope-Async": "enable",
        }
    data = {
            "model": "wanx-background-generation-v2",
            "input":{
                "base_image_url": base_image,
                'ref_image_url':ref_img,
                "ref_prompt": prompt,
            },
            "parameters": {
                "ref_prompt_weight": ref_prompt_weight,
                "n": BATCH_SIZE
            }
        }
    url_create_task = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/background-generation/generation'
    res_ = requests.post(url_create_task, data=json.dumps(data), headers=headers)

    respose_code = res_.status_code
    if 200 == respose_code:
            res = json.loads(res_.content.decode())
            request_id = res['request_id']
            task_id = res['output']['task_id']
            logger.info(f"task_id: {task_id}: Create Background Generation request success. Params: {data}")

            # 异步查询
            is_running = True
            while is_running:
                url_query = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
                res_ = requests.post(url_query, headers=headers)
                respose_code = res_.status_code
                if 200 == respose_code:
                    res = json.loads(res_.content.decode())
                    if "SUCCEEDED" == res['output']['task_status']:
                        logger.info(f"task_id: {task_id}: Background generation task query success.")
                        results = res['output']['results']
                        img_urls = [x['url'] for x in results]
                        logger.info(f"task_id: {task_id}: {res}")
                        break
                    elif "FAILED" != res['output']['task_status']:
                        logger.debug(f"task_id: {task_id}: query result...")
                        time.sleep(1)
                    else:
                        raise gr.Error('Fail to get results from Background Generation task.')

                else:
                    logger.error(f'task_id: {task_id}: Fail to query task result: {res_.content}')
                    raise gr.Error("Fail to query task result.")

            logger.info(f"task_id: {task_id}: download generated images.")
            img_data = download_images(img_urls, BATCH_SIZE)
            logger.info(f"task_id: {task_id}: Generate done.")
            return img_data
    else:
            logger.error(f'Fail to create Background Generation task: {res_.content}')
            raise gr.Error("Fail to create Background Generation task.")
        
