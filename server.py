import disnake
from disnake.ext import commands
from disnake import TextInputStyle
import io
from io import BytesIO, StringIO
from PIL import Image, ImageOps
import asyncio
import queue
import time
import random
import requests
import json
import os
import shlex

ready = True
output = None

from dreams_thread import submit_dream, queues, counter
from datetime import timedelta
from commands import get_command_parser
from modals import PromptModal, ImageModal, InpaintingModal, RowButtons, VisibleRowButtons   

def parse(message):
    try:
        command_parser = get_command_parser()
        opts = command_parser.parse_known_args(shlex.split(message))
        return opts
    except Exception as e:
        print(e)
        return None


class PersistentViewBot(commands.InteractionBot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.persistent_views_added = False

    async def on_ready(self):
        # `on_ready` can be fired multiple times during a bot's lifetime,
        # we only want to register the persistent view once.
        if not self.persistent_views_added:
            # Register the persistent view for listening here.
            # Note that this does not send the view to any message.
            # In order to do this you need to first send a message with the View, which is shown below.
            # If you have the message_id you can also pass it as a keyword argument, but for this example
            # we don't have one.
            self.add_view(VisibleRowButtons())
            self.persistent_views_added = True

        print(f"Logged in as {self.user} (ID: {self.user.id})\n------")

"""
REQUIRED: specify your server ID here
"""
bot = PersistentViewBot(test_guilds=[])


@bot.slash_command(description="bot usage stats")
async def usage(inter):
    running_time = str(timedelta(seconds=time.time() - counter['time']))
    await inter.response.send_message(f"{counter['count']} request, running time: {running_time}", ephemeral=True)


@bot.slash_command(description="Sanity test for prompts")
async def test_prompt(inter, message: str):
    options = parse(message)
    embed = disnake.Embed(title="Prompt Settings")
    for key, value in vars(options[0]).items():
        
        embed.add_field(
            name=key.capitalize(),
            value=value,
            inline=key != "prompt",
        )
    view = RowButtons()
    await inter.response.send_message(view=view, embed=embed, ephemeral=True)

def parse_from_options(options, parameters):
    sizes = {"square": (512,512), "portrait": (512,768), "landscape": (768,512)}

    for key, value in parameters.items():
        if key == 'steps':
            options[key] = min(int(value), 50)
        if key == 'denoising_strength':
            options[key] = min(float(value), 0.99)
        elif key == 'size':
            if value not in ["square", "portrait", "landscape"]:
                value = "square"
            options["width"] = sizes[value][0]
            options["height"] = sizes[value][1]
        elif key == 'cfg_scale':
            options[key] = max(float(value), 1.1)
        elif key == 'sampler_index':
            if value not in ["Euler", "Euler a", "DDIM"]:
                options[key] = "Euler"
            else:
                options[key] = value
        else:
            options[key] = value
    return options



@bot.slash_command(description="Power user prompt command")
async def generate(inter: disnake.AppCmdInter, 
        prompt: str, 
        negative_prompt:str = None,
        size: str = "square",
        steps: int = 20,
        sampler: str = "Euler a",
        seed: int = -1,
        cfg_scale: float = 7.5,
        denoising_strength: float = 0.6,
        init_image: disnake.Attachment = None,
        init_mask: disnake.Attachment = None
    ):
    options = parse("")
    options = vars(options[0])
    counter['count'] += 1 
    options = parse_from_options(options, {"prompt": prompt, "negative_prompt": negative_prompt, "size": size, "steps": steps, "sampler_index": sampler, "seed": seed, "cfg_scale": cfg_scale, "init_image": init_image, "init_mask": init_mask, "denoising_strength": denoising_strength})


    embed = disnake.Embed(title="Prompt Settings") 
    embed.add_field(name="Author", value=inter.author.display_name, inline=False)
    view = RowButtons()

    if inter.channel_id not in queues:
        await inter.response.send_message("Error - unknown channel", ephemeral=True)
    else:
        await inter.response.send_message(f"queued!, position: {len(queues[inter.channel_id])}", ephemeral=True)
        await submit_dream(inter.channel_id, {"inter": inter, "opts": options, "embed": embed, "view": view})
   
@bot.slash_command(description="Run Prompt in Stable Diffusion")
async def prompts(inter: disnake.AppCmdInter):
    """Sends a Modal to create a tag."""
    await inter.response.send_modal(modal=PromptModal())
