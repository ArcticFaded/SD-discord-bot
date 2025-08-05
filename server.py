import disnake
from disnake.ext import commands
from disnake import TextInputStyle
import io
from io import BytesIO, StringIO
from PIL import Image, ImageOps
import asyncio
import time
import random
import requests
import json
import os
import shlex
from temporal_client import temporal_client
from share_cache import share_cache
from datetime import timedelta
from commands import get_command_parser
from modals import PromptModal, RowButtons, VisibleRowButtons   
import json

config = json.load(open("config.json"))

# Statistics tracking
counter = {
    "time": time.time(),
    "count": 0
}

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
        
        # Initialize Temporal client
        await temporal_client.connect()

        print(f"Logged in as {self.user} (ID: {self.user.id})\n------")

"""
REQUIRED: specify your server ID here
"""
bot = PersistentViewBot(test_guilds=config.get("server", []))


@bot.slash_command(description="bot usage stats")
async def usage(inter):
    running_time = str(timedelta(seconds=time.time() - counter['time']))
    
    # Get queue status from Temporal
    queue_status = await temporal_client.get_queue_status(str(inter.channel_id))
    if queue_status:
        queue_info = f", Queue: {queue_status.get('queue_length', 0)} pending"
    else:
        queue_info = ""
    
    await inter.response.send_message(
        f"{counter['count']} requests, running time: {running_time}{queue_info}", 
        ephemeral=True
    )


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
    sizes = {"square": (config["image"]['min_width'], config["image"]['min_height']),
             "portrait": (config["image"]['min_width'], config["image"]['max_height']),
             "landscape": (config["image"]['max_width'], config["image"]['min_height'])}

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


async def submit_to_temporal(inter: disnake.AppCmdInter, options: dict, embed: disnake.Embed):
    """Submit request to Temporal workflow"""
    # Prepare embed data for reconstruction
    embed_data = {
        "title": embed.title,
        "fields": [{"name": field.name, "value": field.value, "inline": field.inline} 
                   for field in embed.fields]
    }
    
    # Get the followup URL for responses
    followup_url = f"https://discord.com/api/webhooks/{inter.application_id}/{inter.token}"
    
    # Submit to Temporal - channel validation happens in the workflow
    success = await temporal_client.submit_image_request(
        channel_id=str(inter.channel_id),
        followup_url=followup_url,
        author_name=inter.author.display_name,
        author_id=str(inter.author.id),
        options=options,
        embed_data=embed_data
    )
    
    if success:
        counter['count'] += 1
        return True
    else:
        await inter.followup.send("❌ Error submitting request. Please try again.", ephemeral=True)
        return False


@bot.slash_command(description="Power user prompt command")
async def generate(inter: disnake.AppCmdInter, 
        prompt: str, 
        negative_prompt:str = None,
        size: str = "square",
        steps: int = config['default_steps'],
        sampler: str = config['default_sampler'],
        seed: int = -1,
        cfg_scale: float = config['default_cfg'],
        denoising_strength: float = 0.6,
        init_image: disnake.Attachment = None,
        init_mask: disnake.Attachment = None
    ):
    options = parse("")
    options = vars(options[0])
    
    options = parse_from_options(options, {
        "prompt": prompt, 
        "negative_prompt": negative_prompt, 
        "size": size, 
        "steps": steps, 
        "sampler_index": sampler, 
        "seed": seed, 
        "cfg_scale": cfg_scale, 
        "init_image": init_image, 
        "init_mask": init_mask, 
        "denoising_strength": denoising_strength
    })

    embed = disnake.Embed(title="Prompt Settings") 
    embed.add_field(name="Author", value=inter.author.display_name, inline=False)
    
    # Respond immediately
    await inter.response.defer(ephemeral=True)
    
    # Submit to Temporal
    await submit_to_temporal(inter, options, embed)

   
@bot.slash_command(description="Run Prompt in Stable Diffusion")
async def prompts(inter: disnake.AppCmdInter):
    """Sends a Modal to create a tag."""
    await inter.response.send_modal(modal=PromptModal())


@bot.slash_command(description="Show active workflow channels")
async def active_channels(inter: disnake.AppCmdInter):
    """Show which channels have active Temporal workflows"""
    channels = await temporal_client.get_active_channels()
    if channels:
        channel_list = "\n".join([f"• <#{ch_id}> → {wf_id}" for ch_id, wf_id in channels.items()])
        embed = disnake.Embed(
            title="Active Temporal Workflows",
            description=channel_list or "No active workflows",
            color=disnake.Color.green()
        )
    else:
        embed = disnake.Embed(
            title="Active Temporal Workflows",
            description="Could not retrieve workflow information",
            color=disnake.Color.red()
        )
    
    await inter.response.send_message(embed=embed, ephemeral=True)

@bot.slash_command(description="Show sharing cache statistics")
async def cache_stats(inter: disnake.AppCmdInter):
    """Show statistics about shared prompts and threads"""
    stats = share_cache.get_channel_stats(str(inter.channel_id))
    
    embed = disnake.Embed(
        title="📊 Share Cache Statistics",
        description=f"Stats for this channel (last {stats['window_hours']} hours)",
        color=disnake.Color.blue()
    )
    
    embed.add_field(
        name="Unique Prompts",
        value=str(stats['unique_prompts']),
        inline=True
    )
    embed.add_field(
        name="Total Shares", 
        value=str(stats['total_shares']),
        inline=True
    )
    embed.add_field(
        name="Threads Created",
        value=str(stats['threads_created']),
        inline=True
    )
    
    if stats['unique_prompts'] > 0:
        avg_shares = stats['total_shares'] / stats['unique_prompts']
        embed.add_field(
            name="Avg Shares/Prompt",
            value=f"{avg_shares:.1f}",
            inline=True
        )
    
    embed.set_footer(text="Cache auto-expires entries after 24 hours")
    
    await inter.response.send_message(embed=embed, ephemeral=True)