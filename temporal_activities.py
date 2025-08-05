from temporalio import activity
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import io
import base64
import json
import aiohttp
from PIL import Image, PngImagePlugin
import uuid
from models import txt2img, img2img
from prompts import fluff_prompt
from cache_prompt import put_prompt
import re
import disnake
from PIL import ImageOps
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "backend"))
from backend.client import DiffSynthClient

@dataclass
class ImageGenerationRequest:
    channel_id: str
    inter_followup_url: str
    author_name: str
    options: Dict[str, Any]
    embed_data: Dict[str, Any]

@dataclass
class ImageGenerationResult:
    success: bool
    image_data: Optional[bytes] = None
    image_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    embed_data: Optional[Dict[str, Any]] = None

@activity.defn
async def generate_image(request: ImageGenerationRequest) -> ImageGenerationResult:
    """Activity to generate an image using Stable Diffusion API"""
    try:
        options = request.options
        
        # Process prompts
        if "negative_prompt" not in options:
            prompt = options["prompt"]
            negative = re.findall(r"\[(.*?)\]", prompt)
            negative = " ".join(negative).strip()
            positive = re.sub(r"\[(.*?)\]", "", prompt).strip()
            options["prompt"] = positive
            options["negative_prompt"] = negative
        
        options["prompt"], options["restore_faces"] = fluff_prompt(options["prompt"], True)
        options["negative_prompt"], _ = fluff_prompt(options["negative_prompt"])
        
        # Get server config for this channel
        config = json.load(open("config.json"))
        channel_config = None
        for cfg in config.get("channels", []):
            if cfg["id"] == request.channel_id:
                channel_config = cfg
                break
        
        if not channel_config:
            return ImageGenerationResult(
                success=False,
                error=f"This channel is not configured for image generation. Please use a configured channel."
            )
        
        # Determine backend type
        backend_type = channel_config.get("backend_type", "sdwebui")  # Default to sdwebui for compatibility
        
        # Generate image based on backend type
        if backend_type == "diffsynth":
            # Use DiffSynth backend
            client = DiffSynthClient(channel_config["api"])
            
            # Map model type from config
            model_type = channel_config.get("model_type", "sd")
            
            # Prepare init image if provided
            init_image_bytes = None
            if "init_image" in options and options["init_image"]:
                if isinstance(options["init_image"], str):
                    # Download from URL
                    async with aiohttp.ClientSession() as session:
                        async with session.get(options["init_image"]) as resp:
                            init_image_bytes = await resp.read()
                else:
                    # Download from attachment
                    async with aiohttp.ClientSession() as session:
                        async with session.get(options["init_image"].url) as resp:
                            init_image_bytes = await resp.read()
            
            # Generate using DiffSynth
            result = await client.generate(
                prompt=options["prompt"],
                negative_prompt=options.get("negative_prompt", ""),
                model_type=model_type,
                width=options.get("width", 512),
                height=options.get("height", 512),
                steps=options.get("steps", 20),
                cfg_scale=options.get("cfg_scale", 7.5),
                seed=options.get("seed", -1),
                sampler=options.get("sampler_index", "euler_a"),
                init_image=init_image_bytes,
                denoising_strength=options.get("denoising_strength", 0.75)
            )
            
            # Format response like SD WebUI
            images = [result["image"]]
            parameters = result["parameters"]
            parameters["seed"] = result["seed"]
            
            # Create pnginfo
            pnginfo = {
                "seed": result["seed"],
                "infotexts": [f"{options['prompt']}\nSteps: {options.get('steps', 20)}, Sampler: {options.get('sampler_index', 'euler_a')}, CFG scale: {options.get('cfg_scale', 7.5)}, Seed: {result['seed']}, Size: {options.get('width', 512)}x{options.get('height', 512)}"]
            }
        else:
            # Use SD WebUI backend (existing code)
            options["server_"] = channel_config["api"]
            
            if "init_image" in options and options["init_image"]:
                images, parameters, pnginfo = await img2img(**options)
            else:
                # Clean up options for txt2img
                if "init_image" in options:
                    del options["init_image"]
                if "init_mask" in options:
                    del options["init_mask"]
                if "denoising_strength" in options:
                    del options["denoising_strength"]
                images, parameters, pnginfo = await txt2img(**options)
        
        # Process the first image
        image_name = f'{str(uuid.uuid4())}.png'
        arr = io.BytesIO()
        
        # Prepare PNG metadata
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
        
        # Save image with metadata
        images[0].save(arr, format='PNG', pnginfo=pnginfo_data)
        arr.seek(0)
        put_prompt(image_name, pngcache)
        
        return ImageGenerationResult(
            success=True,
            image_data=arr.getvalue(),
            image_name=image_name,
            parameters=parameters,
            embed_data=request.embed_data
        )
    
    except Exception as e:
        activity.logger.error(f"Error generating image: {e}")
        return ImageGenerationResult(
            success=False,
            error=str(e)
        )

@activity.defn
async def send_discord_response(
    followup_url: str,
    result: ImageGenerationResult,
    ephemeral: bool = True
) -> bool:
    """Activity to send the generated image back to Discord"""
    try:
        import aiohttp
        
        # Prepare the Discord webhook payload
        form_data = aiohttp.FormData()
        
        if result.success and result.image_data:
            # Add the image file
            form_data.add_field(
                'files[0]',
                result.image_data,
                filename=result.image_name,
                content_type='image/png'
            )
            
            # Prepare embed
            embed = {
                "title": result.embed_data.get("title", "Generated Image"),
                "fields": result.embed_data.get("fields", []),
                "image": {"url": f"attachment://{result.image_name}"}
            }
            
            payload = {
                "embeds": [embed],
                "flags": 64 if ephemeral else 0  # 64 = ephemeral message
            }
            
            form_data.add_field('payload_json', json.dumps(payload))
        else:
            # Error response
            payload = {
                "content": f"❌ Error generating image: {result.error}",
                "flags": 64 if ephemeral else 0
            }
            form_data.add_field('payload_json', json.dumps(payload))
        
        # Send to Discord
        async with aiohttp.ClientSession() as session:
            async with session.post(followup_url, data=form_data) as response:
                if response.status not in [200, 204]:
                    error_text = await response.text()
                    activity.logger.error(f"Discord response failed: {response.status} - {error_text}")
                    return False
                return True
                
    except Exception as e:
        activity.logger.error(f"Error sending Discord response: {e}")
        return False

@activity.defn
async def send_discord_status(
    followup_url: str,
    message: str,
    embed: Optional[Dict] = None,
    ephemeral: bool = True
) -> bool:
    """Activity to send status updates to Discord"""
    try:
        import aiohttp
        
        payload = {
            "content": message,
            "flags": 64 if ephemeral else 0  # 64 = ephemeral message
        }
        
        if embed:
            payload["embeds"] = [embed]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                followup_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status not in [200, 204]:
                    error_text = await response.text()
                    activity.logger.error(f"Discord status update failed: {response.status} - {error_text}")
                    return False
                return True
                
    except Exception as e:
        activity.logger.error(f"Error sending Discord status: {e}")
        return False

@activity.defn
async def check_user_request_limit(channel_id: str, user_id: str, limit: int = 2) -> bool:
    """Check if user has too many pending requests"""
    # This will be handled by Temporal's workflow state
    # For now, always return True (allowed)
    return True