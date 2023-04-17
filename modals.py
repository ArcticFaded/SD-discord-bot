import disnake
from disnake.ext import commands
from disnake.enums import ButtonStyle
from disnake import TextInputStyle
import requests
import shlex
from PIL import Image, PngImagePlugin
from io import BytesIO
from commands import get_command_parser
from dreams_thread import submit_dream, queues, counter, models_
import time
import os
from typing import Optional
import sqlite3
from db_utils import save_generation 
import re
import uuid
from cache_prompt import get_prompt, put_prompt
import json

def parse(message):
    try:
        command_parser = get_command_parser()
        opts = command_parser.parse_known_args(shlex.split(message))
        return opts
    except Exception as e:
        print(e)
        return None
    
def extract_for_processing(embed):
    options = {}
    for field in embed:
        key = field.name.lower()
        value = field.value

        options[key] = value
    return options

def extract_from_pnginfo(pnginfo, embed):
    if pnginfo is None or len(pnginfo) == 0:
        pnginfo = get_prompt(embed.image.url.split("/")[-1])
    parameters = pnginfo.get("parameters", None)
    init_image = pnginfo.get("init_image", None)
    init_mask = pnginfo.get("init_mask", None)

    options = parse("")
    options = vars(options[0])

    if parameters is None:
        options = extract_from_embeds(embed.fields)
        return options, options['seed'] 

    png_options = parameters.split("\n")
    
    if init_image is not None:
        options["init_image"] = init_image
    if init_mask is not None:
        options["init_mask"] = init_mask

    prompt, negative_prompt = "", ""
    if len(png_options) == 2:
        prompt, parameters = png_options
        options["prompt"] = prompt
    else:
        prompt, negative_prompt, parameters = png_options
        options["prompt"] = prompt
        options["negative_prompt"] = negative_prompt[17:]

    fields = re.findall(r"([^:,]*):('[^']*'|[^,']*)", parameters)
    fields = {"_".join(key.lower().strip().split()): value.strip() for (key,value) in fields}

    for key, value in fields.items():
        if key == 'sampler':
            if value != 'None':
                options['sampler_index'] = value
            else:
                options['sampler_index'] = 'Euler a'
        elif key == 'denoising_strength':
            if value != 'None':
                options[key] = min(float(value), 0.99)
        elif key == 'steps':
            options[key] = int(value)
        elif key == 'cfg_scale':
            options[key] = max(float(value), 1.1)
        elif key == 'size':
            width, height = value.split("x")
            options["width"] = min(int(width), 768)
            options["height"] = min(int(height), 768)
    return options, fields['seed']

def extract_from_embeds(embed):
    options = parse("")
    options = vars(options[0])
    
    for field in embed:
        key = field.name.lower()
        value = field.value
        
        if key == 'sampler_index':
            if value != 'None':
                options['sampler_index'] = value
            else:
                options['sampler_index'] = "Euler"
        if key == 'denoising_strength':
            if value != 'None':
                options[key] = min(float(value), 0.99)
        if key in options:
            if value != 'None':
                if key == 'steps':
                    options[key] = int(value)
                elif key == 'width'or key == 'height':
                    options[key] = min(int(value), 1024)
                elif key == 'cfg_scale':
                    options[key] = max(float(value), 1.1)
                elif key == 'strength':
                    options[key] = min(float(value), 0.99)
                else:
                    options[key] = value
    options['seed'] = None
    return options
        
class VisibleRowButtons(disnake.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @disnake.ui.button(label="Use-Prompt", style=ButtonStyle.blurple, custom_id="persistent_example:blurple")
    async def first_button(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        attachment = requests.get(inter.message.embeds[0].image.proxy_url).content
        image = Image.open(BytesIO(attachment))
        options, seed = extract_from_pnginfo(image.info, inter.message.embeds[0])

        sizes = { 512: {512: "square", 768: "portrait"}, 768: {512: "landscape"} }
        if "negative_prompt" in options:
            options["prompt"] += f"[{options['negative_prompt']}]"

        if options["width"] not in sizes or options["height"] not in sizes:
            size = "square"
        else:
            size = sizes[options["width"]][options["height"]]

        await inter.response.send_modal(modal=PromptModal(
            steps=options['steps'], cfg_scale=str(options['cfg_scale']), size=size,
            sampler_index=options['sampler_index'], prompt=options['prompt']))

    @disnake.ui.button(label="Show-Prompt", style=ButtonStyle.green, custom_id="persistent_example:green")
    async def second_button(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        attachment = requests.get(inter.message.embeds[0].image.proxy_url).content
        image = Image.open(BytesIO(attachment))
        options, seed = extract_from_pnginfo(image.info, inter.message.embeds[0])

        sizes = { 512: {512: "square", 768: "portrait"}, 768: {512: "landscape"} }
        generate = "/generate "
        for option, value in options.items():
            if option in ["width", "height", "seed", "denoising_strength", "init_image"]:
                continue
            if option == 'sampler_index':
                generate += f"sampler: {value} "
                continue

            generate += f"{option}: {value} "
        if options["width"] not in sizes or options["height"] not in sizes:
            size = "square"
        else:
            size = sizes[options["width"]][options["height"]]
        generate += f"size: {size}"
        await inter.response.send_message(f"`{generate}`\nseed:{seed}", ephemeral=True)

    @disnake.ui.button(label="Delete", style=ButtonStyle.red, custom_id="persistent_example:red")
    async def third_button(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        embed = inter.message.embeds[0]
        author = embed.fields[0].value

        """
        Add your discord ID here, or any moderators to give them override on delete
        """
        if author == inter.author.display_name or inter.author.id in []:
            await inter.message.delete()
        else:
            await inter.response.send_message(f"Please contact a moderator to delete this post.", ephemeral=True)
    
# Defines a simple view of row buttons.
class RowButtons(disnake.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
    
    @disnake.ui.button(label="Re-roll", style=ButtonStyle.blurple)
    async def first_button(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        await inter.response.send_message(f"Re-rolling. position: {len(queues[inter.channel_id])}", ephemeral=True)
        
        counter["count"] += 1
        attachment = requests.get(inter.message.embeds[0].image.proxy_url).content
        image = Image.open(BytesIO(attachment))
        options, seed = extract_from_pnginfo(image.info, inter.message.embeds[0])
        view = RowButtons()

        await submit_dream(inter.channel_id, {"inter": inter, "opts": options, "embed": inter.message.embeds[0], "view": view})

    @disnake.ui.button(label="Use-Prompt", style=ButtonStyle.blurple)
    async def fourth_button(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        # options = extract_from_embeds(inter.message.embeds[0].fields)
        attachment = requests.get(inter.message.embeds[0].image.proxy_url).content
        image = Image.open(BytesIO(attachment))
        options, seed = extract_from_pnginfo(image.info, inter.message.embeds[0])

        sizes = { 512: {512: "square", 768: "portrait"}, 768: {512: "landscape"} }
        if "negative_prompt" in options:
            options["prompt"] += f"[{options['negative_prompt']}]"

        if options["width"] not in sizes or options["height"] not in sizes:
            size = "square"
        else:
            size = sizes[options["width"]][options["height"]]
        # print(options, size)
        await inter.response.send_modal(modal=PromptModal(
            steps=options['steps'], cfg_scale=str(options['cfg_scale']), size=size,
            sampler_index=options['sampler_index'], prompt=options['prompt']))

    @disnake.ui.button(label="Share", style=ButtonStyle.green)
    async def second_button(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        view = VisibleRowButtons()
        image_url = inter.message.embeds[0].image.url
        attachment = requests.get(inter.message.embeds[0].image.url).content
        image = Image.open(BytesIO(attachment))
        options, seed = extract_from_pnginfo(image.info, inter.message.embeds[0])
        embed = inter.message.embeds[0]
        # print(image, len(attachment)) 
        options['author'] = embed.fields[0].value
        # if "init_image" in options and options['init_image'] and options['init_image'] != 'None':
        arr = BytesIO()

        pnginfo_data = PngImagePlugin.PngInfo()
        pnginfo_data.add_text("parameters", image.info['parameters']) 

        image.save(arr, format='PNG', pnginfo=pnginfo_data)
        arr.seek(0)
        image_name = f'{str(uuid.uuid4())}.png'
        put_prompt(image_name, image.info)
        file = disnake.File(fp=arr, filename=image_name)
        embed.set_image(url=f"attachment://{image_name}")
        await inter.response.send_message(file=file, view=view, embed=embed)
       
        """
        After a request is shared we store the information,
        entirely optional comment out if you dont need it
        """
        save_generation(image_url, options["prompt"], options["author"], model)


    @disnake.ui.button(label="Show-Prompt", style=ButtonStyle.green)
    async def fifth_button(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        attachment = requests.get(inter.message.embeds[0].image.proxy_url).content
        image = Image.open(BytesIO(attachment))
        options, seed = extract_from_pnginfo(image.info, inter.message.embeds[0])

        sizes = { 512: {512: "square", 768: "portrait"}, 768: {512: "landscape"} }
        generate = "/generate "
        for option, value in options.items():
            if option in ["width", "height", "seed", "denoising_strength", "init_image"]:
                continue
            if option == 'sampler_index':
                generate += f"sampler: {value} "
                continue

            generate += f"{option}: {value} "
        if options["width"] not in sizes or options["height"] not in sizes:
            size = "square"
        else:
            size = sizes[options["width"]][options["height"]]
        generate += f"size: {size}"
        await inter.response.send_message(f"`{generate}`\nseed:{seed}", ephemeral=True) 

    @disnake.ui.button(label="Delete", style=ButtonStyle.red)
    async def third_button(self, button: disnake.ui.Button, inter: disnake.MessageInteraction): 
        await inter.response.edit_message("deleted", embed=None, attachments=None, view=None)
    
class PromptModal(disnake.ui.Modal):
    def __init__(self, steps=None, cfg_scale=None, size=None, sampler_index=None, prompt=None):
        # The details of the modal, and its components
        components = [ # not exposing strength for now
            disnake.ui.TextInput(
                label="Steps - Max 50",
                placeholder="20",
                value=steps,
                custom_id="steps",
                style=TextInputStyle.short,
                max_length=3,
                required=False
            ),
            disnake.ui.TextInput(
                label="CFG Guidence",
                placeholder="7.5",
                value=cfg_scale,
                custom_id="cfg_scale",
                style=TextInputStyle.short,
                max_length=5,
                required=False
            ),
            disnake.ui.TextInput(
                label="Select a resolution - default square",
                value=size or "square|portrait|landscape",
                placeholder="square",
                style=TextInputStyle.short,
                custom_id="size"
            ),
            disnake.ui.TextInput(
                label="Select one sampler  - default Euler",
                value=sampler_index or "Euler|Euler a|DDIM",
                placeholder="Euler",
                style=TextInputStyle.short,
                custom_id="sampler_index",
            ),
            disnake.ui.TextInput(
                label="Prompt",
                value=prompt,
                placeholder="Prompt goes here.",
                custom_id="prompt",
                style=TextInputStyle.paragraph,
            ),
        ]
        
        super().__init__(
            title="Create Prompt",
            custom_id="create_prompt",
            components=components,
        )

    # The callback received when the user input is completed.
    async def callback(self, inter: disnake.ModalInteraction):
        embed = disnake.Embed(title="Prompt Creation")
        options = parse("")
        options = vars(options[0])

        sizes = {"square": (512,512), "portrait": (512,768), "landscape": (768,512)} 
        
        for key, value in inter.text_values.items():
            if value == "":
                value = "Unknown" if key not in options else str(options[key])
            else:
                if key == "prompt":
                    options[key] = value
                elif key == "size":
                    if value not in ["square", "portrait", "landscape"]:
                        value = "square"
                    options["width"] = sizes[value][0]
                    options["height"] = sizes[value][1]
                elif key == "sampler_index":
                    if value not in ["Euler", "Euler a", "DDIM"]:
                        options[key] = "Euler"
                    else:
                        options[key] = value
                elif key == "steps":
                    try:
                        options[key] = min(int(value), 1024)
                    except ValueError:
                            pass
                elif key == "width" or key == "height":
                    try:
                        options[key] = min(int(value), 1024)
                    except ValueError:
                            pass
                else:
                    try:
                        options[key] = float(value)
                        if key == "strength":

                                options[key] = min(float(value), 0.99)
                    except ValueError:
                        pass

        embed.add_field(name="Author", value=inter.author.display_name, inline=False)
        view = RowButtons()
        # print(options)
        if inter.channel_id not in queues:
            await inter.response.send_message("Unknown Channel", ephemeral=True)
        else:
            await inter.response.send_message(f"queued!, position: {len(queues[inter.channel_id])}", ephemeral=True)
            counter["count"] += 1 
            await submit_dream(inter.channel_id, {"inter": inter, "opts": options, "embed": embed, "view": view})
        
        
