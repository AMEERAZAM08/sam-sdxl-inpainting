import random

import numpy as np
import cv2
import os
import io
import oss2
from PIL import Image

import dashscope
from dashscope import MultiModalConversation

from http import HTTPStatus
import re
import requests
from .log import logger
import concurrent.futures

# dashscope.api_key = os.getenv("API_KEY_QW")
# oss
# access_key_id = os.getenv("ACCESS_KEY_ID")
# access_key_secret = os.getenv("ACCESS_KEY_SECRET")
# bucket_name = os.getenv("BUCKET_NAME")
# endpoint = os.getenv("ENDPOINT")

# bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
oss_path = "ashui"
oss_path_img_gallery = "ashui_img_gallery"

def download_img_pil(index, img_url):
    # print(img_url)
    r = requests.get(img_url, stream=True)
    if r.status_code == 200:
        img = Image.open(io.BytesIO(r.content))
        return (index, img)
    else:
        logger.error(f"Fail to download: {img_url}")


def download_images(img_urls, batch_size):
    imgs_pil = [None] * batch_size
    # worker_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        to_do = []
        for i, url in enumerate(img_urls):
            future = executor.submit(download_img_pil, i, url)
            to_do.append(future)

        for future in concurrent.futures.as_completed(to_do):
            ret = future.result()
            # worker_results.append(ret)
            index, img_pil = ret
            imgs_pil[index] = img_pil  # 按顺序排列url，后续下载关联的图片或者svg需要使用

    return imgs_pil

def upload_np_2_oss(input_image, name="cache.png", gallery=False):
    imgByteArr = io.BytesIO()
    Image.fromarray(input_image).save(imgByteArr, format="PNG")
    imgByteArr = imgByteArr.getvalue()

    if gallery:
        path = oss_path_img_gallery
    else:
        path = oss_path

    # bucket.put_object(path+"/"+name, imgByteArr)  # data为数据，可以是图片
    # ret = bucket.sign_url('GET', path+"/"+name, 60*60*24)  # 返回值为链接，参数依次为，方法/oss上文件路径/过期时间(s)
    del imgByteArr
    return ""


def call_with_messages(prompt):
    messages = [
        {'role': 'user', 'content': prompt}]
    response = dashscope.Generation.call(
        'qwen-14b-chat',
        messages=messages,
        result_format='message',  # set the result is message format.
    )
    if response.status_code == HTTPStatus.OK:
        return response['output']["choices"][0]["message"]['content']
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        return None

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


def min_max_norm(x):
    x -= np.min(x)
    x /= np.maximum(np.max(x), 1e-5)
    return x


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def img2mask(img, H, W, low=10, high=90):
    assert img.ndim == 3 or img.ndim == 2
    assert img.dtype == np.uint8

    if img.ndim == 3:
        y = img[:, :, random.randrange(0, img.shape[2])]
    else:
        y = img

    y = cv2.resize(y, (W, H), interpolation=cv2.INTER_CUBIC)

    if random.uniform(0, 1) < 0.5:
        y = 255 - y

    return y < np.percentile(y, random.randrange(low, high))
