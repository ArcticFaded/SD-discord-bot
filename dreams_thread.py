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

def create_image(options, inter, event_loop):
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
    
    options["prompt"] = options["prompt"].replace("teen", "")
    options["prompt"] = options["prompt"].replace("young", "")

    """
    The server also supports using webui with img2img
    """
    if "init_image" in options and options["init_image"]:
        return img2img(**(options), event_loop=event_loop)
    else:
        if "init_image" in options: 
            if "init_image" in options: del options["init_image"]
            if "init_mask" in options: del options["init_mask"]
            if "denoising_strength" in options: del options["denoising_strength"]
        return txt2img(**(options))


async def submit_dream(channel_id, queue_object):
    if len(queues[channel_id]) > 0:
        """
        Optionally control how many request a user can make, in this case they are limited to at most 2 requests at a queue
        """
        users = [t['inter'].author.id for t in queues[channel_id]]
        if queue_object['inter'].author.id in users:
            await queue_object['inter'].followup.send("you already have a work in the queue!", ephemeral=True)
            return

    if threads[channel_id] and threads[channel_id].is_alive():
        queues[channel_id].append(queue_object)
    else:
        threads[channel_id] = Thread(target=dream, args=(event_loop, queue_object)) 
        threads[channel_id].start()

def dream(event_loop, queue_object):
    inter = queue_object["inter"]

    if queue_object["opts"] is None:
        event_loop.task(inter.followup.send("Error when parsing your request"))
    else:
        options = queue_object["opts"]
        embed = queue_object["embed"]
        view = queue_object["view"]
        options["server_"] = server_[inter.channel_id]
        images, parameters, pnginfo = create_image(options, inter, event_loop)
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
        event_loop.create_task(inter.followup.send(file=file, view=view, embeds=embeds, ephemeral=True))
        # event_loop.create_task(inter.followup.send(files=files, view=view, embeds=embeds, ephemeral=True))
    if queues[inter.channel_id]:
        event_loop.create_task(submit_dream(inter.channel_id, queues[inter.channel_id].pop(0)))

