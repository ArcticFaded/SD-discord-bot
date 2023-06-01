from threading import Thread
import asyncio
from models import txt2img, img2img
import io
import re
import disnake
import time
from PIL import PngImagePlugin
import uuid
from PIL import Image, ImageOps
import json
from cache_prompt import put_prompt

from prompts import fluff_prompt

config = json.load(open("config.json"))
counter = { 
    "time": time.time(), 
    "count": 0 
}

"""
models_ is a mapping from channel id -> model name
the assumption here is that each instance of webui is
dedicated to one model
Example:
{
    1231412412: "NAI"
}

Used for the purpose of tagging rich metadata
"""
models_ = {cfg["id"]: cfg["model"] for cfg in config.get("channels", [])}
"""
servers_ is a mapping from channel id -> webui server
Example:
{
    1231412412: "http://localhost:7860"
}
"""
server_ = {cfg["id"]: cfg["api"] for cfg in config.get("channels", [])}

queues = {item: [] for item in server_}
threads = {item: None for item in server_}
event_loop = asyncio.get_event_loop()

apply_watermark = False
"""
Apply watermarks to your generations if needed
"""
if apply_watermark:
    watermark = Image.open("watermark.png")
    resized = ImageOps.contain(watermark, (64,64))
    resized = resized.rotate(90*3, Image.NEAREST, expand = 1)

def add_watermark(image):
    if apply_watermark:
        size = image.size
        image.paste(resized, (size[0]-32,size[1]-128), resized)
    return image

async def create_image(options, inter):
    """
    This parsing code is to allow for flexable usage of negative prompts between
    /generate and /prompts

    To use a negative prompt in the model interface just place the words between [this]
    """
    if "negative_prompt" not in options:
        prompt = options["prompt"]
        negative = re.findall(r"\[(.*?)\]", prompt)
        negative = " ".join(negative).strip()
        positive = re.sub(r"\[(.*?)\]", "", prompt).strip()
                     
        options["prompt"] = positive
        options["negative_prompt"] = negative
    
    options["prompt"], options["restore_faces"] = fluff_prompt(options["prompt"], True)
    options["negative_prompt"], _ = fluff_prompt(options["negative_prompt"])
    
    """
    The server also supports using webui with img2img
    """
    if "init_image" in options and options["init_image"]:
        return await img2img(**(options))
    else:
        if "init_image" in options: 
            if "init_image" in options: del options["init_image"]
            if "init_mask" in options: del options["init_mask"]
            if "denoising_strength" in options: del options["denoising_strength"]
        return await txt2img(**(options))


async def submit_dream(channel_id, queue_object):
    if len(queues[channel_id]) > 0:
        """
        Optionally control how many request a user can make, in this case they are limited to at most 2 requests at a queue
        """
        users = [t['inter'].author.id for t in queues[channel_id]]
        if queue_object['inter'].author.id in users:
            await queue_object['inter'].followup.send("you already have a work in the queue!", ephemeral=True)
            return

    if threads[channel_id] and not threads[channel_id].done():
        queues[channel_id].append(queue_object)
    else:
        threads[channel_id] = asyncio.create_task(dream(queue_object))

async def dream(queue_object):
    inter = queue_object["inter"]

    if queue_object["opts"] is None:
        await inter.followup.send("Error when parsing your request")
    else:
        options = queue_object["opts"]
        embed = queue_object["embed"]
        view = queue_object["view"]
        options["server_"] = server_[inter.channel_id]
        images, parameters, pnginfo = await create_image(options, inter)
        embeds = [embed] 
        seed = parameters['seed']
        image_name = f'{str(uuid.uuid4())}.png'
            
        seed = parameters['seed']
        arr = io.BytesIO()

        pngcache = {}
        pnginfo_data = PngImagePlugin.PngInfo()
        pnginfo_data.add_text("parameters", pnginfo["infotexts"][0])
        pngcache["parameters"] = pnginfo["infotexts"][0]
        if "init_image" in options and options["init_image"]:
            if isinstance(options['init_image'], str):
                pngcache["init_image"] = options["init_image"]
                pnginfo_data.add_text("init_image", options["init_image"])
            else:
                pngcache["init_image"] = options["init_image"].url
                pnginfo_data.add_text("init_image", options["init_image"].url)
        if "init_mask" in options and options["init_mask"]:
            if isinstance(options['init_mask'], str):
                pngcache["init_mask"] = options["init_mask"]
                pnginfo_data.add_text("init_mask", options["init_mask"])
            else:
                pngcache["init_mask"] = options["init_mask"].url
                pnginfo_data.add_text("init_mask", options["init_mask"].url)
        """
        images is an array because it is possible to generate multiple images at once in a
        single request, but the example shown here does it for one image

        add_watermark only works if watermark=True
        """
        images[0] = add_watermark(images[0])
        images[0].save(arr, format='PNG', pnginfo=pnginfo_data)
        arr.seek(0)
        put_prompt(image_name, pngcache)
        file = disnake.File(fp=arr, filename=image_name)
        embeds[0].set_image(url=f"attachment://{image_name}")
        await inter.followup.send(file=file, view=view, embeds=embeds, ephemeral=True)
        # event_loop.create_task(inter.followup.send(files=files, view=view, embeds=embeds, ephemeral=True))
    if queues[inter.channel_id]:
        asyncio.create_task(dream(queues[channel_id].pop(0)))
 
