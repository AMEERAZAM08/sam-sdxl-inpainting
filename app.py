##!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023-06-01
# @Author  : ashui(Binghui Chen)
from sympy import im
import time
import cv2
import gradio as gr
import numpy as np
import random
import math
import uuid
import torch
from torch import autocast

from src.util import resize_image, upload_np_2_oss
from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
import diffusers
import sys, os

from PIL import Image, ImageFilter, ImageOps, ImageDraw

from segment_anything import SamPredictor, sam_model_registry


device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)

mobile_sam = sam_model_registry['vit_h'](checkpoint='models/sam_vit_h_4b8939.pth').to("cuda")
mobile_sam.eval()
mobile_predictor = SamPredictor(mobile_sam)
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

# - - - - - examples  - - - - -  #
# 输入图地址, 文本, 背景图地址, index, []
image_examples = [
                            ["imgs/000.jpg", "A young woman in short sleeves shows off a mobile phone", None, 0, []],
                            ["imgs/001.jpg", "A young woman wears short sleeves, her hand is holding a bottle.", None, 1, []],
                            ["imgs/003.png", "A woman is wearing a black suit against a blue background", "imgs/003_bg.jpg", 2, []],
                            ["imgs/002.png", "A young woman poses in a dress, she stands in front of a blue background", "imgs/002_bg.png", 3, []],
                            ["imgs/bg_gen/base_imgs/1cdb9b1e6daea6a1b85236595d3e43d6.png", "water splash", None, 4, []],
                            ["imgs/bg_gen/base_imgs/1cdb9b1e6daea6a1b85236595d3e43d6.png", "", "imgs/bg_gen/ref_imgs/df9a93ac2bca12696a9166182c4bf02ad9679aa5.jpg", 5, []],
                            ["imgs/bg_gen/base_imgs/IMG_2941.png", "On the desert floor", None, 6, []],
                            ["imgs/bg_gen/base_imgs/b2b1ed243364473e49d2e478e4f24413.png","White ground, white background, light coming in, Canon",None,7,[]],
                        ]

img = "image_gallery/"
files = os.listdir(img)
files = sorted(files)
showcases = []
for idx, name in enumerate(files):
        temp = os.path.join(os.path.dirname(__file__), img, name)
        showcases.append(temp)

def process(original_image, original_mask, input_mask, selected_points, prompt,negative_prompt,guidance_scale,steps,strength,scheduler):
    if original_image.shape[0]>original_image.shape[1]:
        original_image=cv2.resize(original_image,(int(original_image.shape[1]*1000/original_image.shape[0]),1000))
    if original_mask.shape[0]>original_mask.shape[1]:
        original_mask=cv2.resize(original_mask,(int(original_mask.shape[1]*1000/original_mask.shape[0]),1000))
    if original_image is None:
        raise gr.Error('Please upload the input image')
    if (original_mask is None or len(selected_points)==0) and input_mask is None:
        raise gr.Error("Please click the region where you want to keep unchanged, or upload a white-black Mask image where white color indicates region to be retained.")
    
    # load example image
    if isinstance(original_image, int):
            image_name = image_examples[original_image][0]
            original_image = cv2.imread(image_name)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if input_mask is not None:
        H,W=original_image.shape[:2]
        original_mask = cv2.resize(input_mask, (W, H))
    else:
        original_mask = np.clip(255 - original_mask, 0, 255).astype(np.uint8)

    request_id = str(uuid.uuid4())
    # input_image_url = upload_np_2_oss(original_image, request_id+".png")
    # input_mask_url = upload_np_2_oss(original_mask, request_id+"_mask.png")
    # source_background_url = "" if source_background is None else upload_np_2_oss(source_background, request_id+"_bg.png")
    if negative_prompt == "":
        negative_prompt = None
    scheduler_class_name = scheduler.split("-")[0]

    add_kwargs = {}
    if len(scheduler.split("-")) > 1:
        add_kwargs["use_karras"] = True
    if len(scheduler.split("-")) > 2:
        add_kwargs["algorithm_type"] = "sde-dpmsolver++"

    scheduler = getattr(diffusers, scheduler_class_name)
    pipe.scheduler = scheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **add_kwargs)
    
    # Image.fromarray(original_mask).save("original_mask.png")
    init_image = Image.fromarray(original_image).convert("RGB")
    mask = Image.fromarray(original_mask).convert("RGB")
    output = pipe(prompt = prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask, guidance_scale=guidance_scale, num_inference_steps=int(steps), strength=strength)
    # person detect: [[x1,y1,x2,y2,score],]
    # det_res = call_person_detect(input_image_url)

    res = []
    # if len(det_res)>0:
    #     if len(prompt)==0:
    #         raise gr.Error('Please input the prompt')
    #     # res = call_virtualmodel(input_image_url, input_mask_url, source_background_url, prompt, face_prompt)
    # else:
    #     ###
    #     if len(prompt)==0:
    #         prompt=None
    #     ref_image_url=None if source_background_url =='' else source_background_url
    #     original_mask=original_mask[:,:,:1]
    #     base_image=np.concatenate([original_image, original_mask],axis=2)
    #     base_image_url=upload_np_2_oss(base_image, request_id+"_base.png")
    #     res=call_bg_genration(base_image_url,ref_image_url,prompt,ref_prompt_weight=0.5)
    # Image.fromarray(input_mask).save("input_mask.png")
    res= output.images[0]
    res = res.convert("RGB")
    #resize the output image to original image size
    res = res.resize((original_image.shape[1],original_image.shape[0]), Image.LANCZOS)
    return [res], request_id, True

block = gr.Blocks(
        css="css/style.css",
        theme=gr.themes.Soft(
             radius_size=gr.themes.sizes.radius_none,
             text_size=gr.themes.sizes.text_md
         )
        ).queue(concurrency_count=2)
with block:
    with gr.Row():
        with gr.Column():
            gr.HTML(f"""
                    </br>
                    <div class="baselayout" style="text-shadow: white 0.01rem 0.01rem 0.4rem; position:fixed; z-index: 9999; top:0; left:0;right:0; background-size:100% 100%">
                        <h1 style="text-align:center; color:Black; font-size:3rem; position: relative;"> SAM + SDXL Inpainting </h1>
                    </div>
                    </br>
                    </br>
                    <div style="text-align: center;">
                        <h1 >ReplaceAnything using SAM + SDXL Inpainting as you want: Ultra-high quality content replacement</h1>
                    </div>
            """)

    with gr.Tabs(elem_classes=["Tab"]):
        with gr.TabItem("Image Create"):  
            with gr.Accordion(label="🧭 Instructions:", open=True, elem_id="accordion"):
                with gr.Row(equal_height=True):
                    gr.Markdown("""
                    - ⭐️ <b>step1：</b>Upload or select one image from Example
                    - ⭐️ <b>step2：</b>Click on Input-image to select the object to be retained (or upload a white-black Mask image, in which white color indicates the region you want to keep unchanged)
                    - ⭐️ <b>step3：</b>Input prompt or reference image (highly-recommended) for generating new contents
                    - ⭐️ <b>step4：</b>Click Run button
                    """)                          
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_id="Input"):
                        with gr.Row():
                            with gr.Tabs(elem_classes=["feedback"]):
                                with gr.TabItem("Input Image"):
                                    input_image = gr.Image(type="numpy", label="input",scale=2)
                        original_image = gr.State(value=None,label="index")
                        original_mask = gr.State(value=None)
                        selected_points = gr.State([],label="click points")
                        with gr.Row(elem_id="Seg"):
                            radio = gr.Radio(['foreground', 'background'], label='Click to seg: ', value='foreground',scale=2)
                            undo_button = gr.Button('Undo seg', elem_id="btnSEG",scale=1)
                    input_mask = gr.Image(type="numpy", label="Mask Image")
                    prompt = gr.Textbox(label="Prompt", placeholder="Please input your prompt",value='',lines=1)
                    negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Please input your prompt",value='hand,blur,face,bad',lines=1)
                    guidance_scale = gr.Number(value=7.5, minimum=1.0, maximum=20.0, step=0.1, label="guidance_scale")  
                    steps = gr.Number(value=20, minimum=10, maximum=30, step=1, label="steps")
                    strength = gr.Number(value=0.99, minimum=0.01, maximum=1.0, step=0.01, label="strength")
                    with gr.Row(mobile_collapse=False, equal_height=True):
                        schedulers = ["DEISMultistepScheduler", "HeunDiscreteScheduler", "EulerDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler-Karras", "DPMSolverMultistepScheduler-Karras-SDE"]
                        scheduler = gr.Dropdown(label="Schedulers", choices=schedulers, value="EulerDiscreteScheduler")

                    run_button = gr.Button("Run",elem_id="btn")
                    
                with gr.Column():
                    with gr.Tabs(elem_classes=["feedback"]):
                        with gr.TabItem("Outputs"):
                            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True)
                            # recommend=gr.Button("Recommend results to Image Gallery",elem_id="recBut")
                            request_id=gr.State(value="")
                            gallery_flag=gr.State(value=False)

     # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        # image upload is too slow
        # if min(img.shape[0], img.shape[1]) > 896:
        #     img = resize_image(img, 896)
        # if max(img.shape[0], img.shape[1])*1.0/min(img.shape[0], img.shape[1])>2.0:
        #     raise gr.Error('image aspect ratio cannot be larger than 2.0')
        return img, img, [], None  # when new image is uploaded, `selected_points` should be empty

    input_image.upload(
        store_img,
        [input_image],
        [input_image, original_image, selected_points]
    )

    # user click the image to get points, and show the points on the image
    def segmentation(img, sel_pix):
        print("segmentation")
        # online show seg mask
        points = []
        labels = []
        for p, l in sel_pix:
            points.append(p)
            labels.append(l)
        mobile_predictor.set_image(img if isinstance(img, np.ndarray) else np.array(img))
        with torch.no_grad():
            with autocast("cuda"):
                masks, _, _ = mobile_predictor.predict(point_coords=np.array(points), point_labels=np.array(labels), multimask_output=False)

        output_mask = np.ones((masks.shape[1], masks.shape[2], 3))*255
        for i in range(3):
                output_mask[masks[0] == True, i] = 0.0

        mask_all = np.ones((masks.shape[1], masks.shape[2], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
                mask_all[masks[0] == True, i] = color_mask[i]
        masked_img = img / 255 * 0.3 + mask_all * 0.7
        masked_img = masked_img*255
        ## draw points
        for point, label in sel_pix:
            cv2.drawMarker(masked_img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        return masked_img, output_mask
    
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        
        if point_type == 'foreground':
            sel_pix.append((evt.index, 1))   # append the foreground_point
        elif point_type == 'background':
            sel_pix.append((evt.index, 0))    # append the background_point
        else:
            sel_pix.append((evt.index, 1))    # default foreground_point

        if isinstance(img, int):
            image_name = image_examples[img][0]
            img = cv2.imread(image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # online show seg mask
        if img.shape[0]>img.shape[1]:
            img=cv2.resize(img,(int(img.shape[1]*1000/img.shape[0]),1000))
        masked_img, output_mask = segmentation(img, sel_pix)
       
        return masked_img.astype(np.uint8), output_mask
    
    input_image.select(
        get_point,
        [original_image, selected_points, radio],
        [input_image, original_mask],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):
        # draw points
        output_mask = None
        if len(sel_pix) != 0:
            if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
                temp = cv2.imread(image_examples[orig_img][0])
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            else:
                temp = orig_img.copy()
            sel_pix.pop()
            # online show seg mask
            if len(sel_pix) !=0:
                temp, output_mask = segmentation(temp, sel_pix)
            return temp.astype(np.uint8), output_mask
        else:
            gr.Error("Nothing to Undo")
    
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image, original_mask]
    )

    def upload_to_img_gallery(img, res, re_id, flag):
        if flag:
            gr.Info("Image uploading")
            if isinstance(img, int):
                image_name = image_examples[img][0]
                img = cv2.imread(image_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _ = upload_np_2_oss(img, name=re_id+"_ori.jpg", gallery=True)
            for idx, r in enumerate(res):
                r = cv2.imread(r['name'])
                r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
                _ = upload_np_2_oss(r, name=re_id+f"_res_{idx}.jpg", gallery=True)
            flag=False
            gr.Info("Images have beend uploaded and are under check")
        else:
            gr.Info("Nothing to to")
        return flag

    # recommend.click(
    #     upload_to_img_gallery,
    #     [original_image, result_gallery, request_id, gallery_flag],
    #     [gallery_flag]
    # )
    # ips=[input_image, original_image, original_mask, input_mask, selected_points, prompt,negative_prompt,guidance_scale,steps,strength,scheduler]
    ips=[original_image, original_mask, input_mask, selected_points, prompt,negative_prompt,guidance_scale,steps,strength,scheduler]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, request_id, gallery_flag])


block.launch(share=True)
