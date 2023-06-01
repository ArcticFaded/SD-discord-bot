from io import BytesIO
import io
import requests
from PIL import Image, ImageOps
import base64
import json
import aiohttp

def extras():
    pass

def extract_image(bytearr):
    image = Image.open(BytesIO(bytearr))
    
    w, h = image.size
    if w > 704 or h > 704:
        resize = ImageOps.contain(image, (704,704))
    else:
        resize = image.copy()
    width, height = resize.size
    width = 64 * int((width + 32) / 64)
    height = 64 * int((height + 32) / 64)

    buffer_ = BytesIO()
    resize.save(buffer_, format="PNG")
    buffer_.seek(0)

    return buffer_, width, height

async def download_attachment(attachment):
    async with aiohttp.ClientSession() as session:
        if isinstance(attachment, str):
            async with session.get(attachment, timeout=15) as resp:
                return await resp.content.read()
        else:
            async with session.get(attachment.url, timeout=15) as resp:
                return await resp.content.read()

async def img2img(prompt: str,
            negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            steps: int = 28,
            sampler_index: str = "Euler",
            restore_faces: bool = False,
            cfg_scale: float = 11,
            height: int = 512,
            width: int = 512,
            seed: int = -1,
            server_: str = "http://127.0.0.1:7860",
            init_image: any = None,
            init_mask: any = None,
            denoising_strength: float = 0.6,
            *args):
    attachment = await download_attachment(init_image)
    mask_attachment = None
    if init_mask:
        mask_attachment = await download_attachment(init_mask)
    image, width, height = extract_image(attachment)
    payload = {
        "prompt":prompt,
        "negative_prompt": negative_prompt,
        "sampler_index": sampler_index,
        "steps": min(steps, 50),
        "cfg_scale": max(cfg_scale, 1),
        "width": min(width, 768),
        "height": min(height, 768),
        "seed": seed,
        "restore_faces": "7860" not in server_ and "7861" not in server_,
        "denoising_strength": denoising_strength,
        "init_images": ["data:image/png;base64," + base64.b64encode(image.getvalue()).decode()]
    }

    if mask_attachment:
        mask_image, _, _ = extract_image(mask_attachment)
        payload["mask"] = "data:image/png;base64," + base64.b64encode(mask_image.getvalue()).decode()

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{server_}/sdapi/v1/img2img", json=payload, timeout=180) as response:
            resp = await response.json()
    
    images = resp["images"]
    for i in range(len(images)):
        if "," in images[i]:
            images[i] = images[i].split(",")[-1]
    parameters = resp["parameters"]
    info = resp["info"]
    pnginfo = json.loads(info)
    # generated_params = list(map(lambda x: x.split(":"), info.split(",")))
    # info_items = {item[0].strip():item[1].strip() for item in generated_params if len(item) > 1}

    parameters["seed"] = int(pnginfo["seed"])
    processed = [Image.open(io.BytesIO(base64.b64decode(image))) for image in images]
    return processed, resp["parameters"], pnginfo


async def txt2img(prompt: str,
            negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry", 
            steps: int = 28, 
            sampler_index: str = "Euler", 
            restore_faces: bool = False, 
            cfg_scale: float = 11, 
            height: int = 512, 
            width: int = 512,
            seed: int = -1,
            server_: str = "http://127.0.0.1:7860",
            *args):

    payload = {
        "prompt":prompt,
        "negative_prompt": negative_prompt,
        #There are tons of optional parms, let's just set the sampler
        "steps": min(steps, 50),
        "sampler_index": sampler_index,
        "cfg_scale": max(cfg_scale, 1),
        "width": min(width, 768),
        "height": min(height, 768),
        "restore_faces": "7860" not in server_ and "7861" not in server_,
        "seed": seed
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{server_}/sdapi/v1/txt2img", json=payload, timeout=180) as response:
            resp = await response.json()
    images = resp["images"]
    for i in range(len(images)):
        if "," in images[i]:
            images[i] = images[i].split(",")[-1] 
    parameters = resp["parameters"]
    info = resp["info"]
    pnginfo = json.loads(info)
    
    # generated_params = list(map(lambda x: x.split(":"), info.split(",")))
    # info_items = {item[0].strip():item[1].strip() for item in generated_params if len(item) > 1}
    parameters["seed"] = int(pnginfo["seed"])
    processed = [Image.open(io.BytesIO(base64.b64decode(image))) for image in images]
    return processed, resp["parameters"], pnginfo
