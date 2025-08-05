"""
Client for DiffSynth backend API
"""

import aiohttp
import base64
import io
from PIL import Image
from typing import Optional, Dict, Any, List

class DiffSynthClient:
    """Client for communicating with DiffSynth backend"""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_type: str = "sd",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: int = -1,
        sampler: str = "euler_a",
        init_image: Optional[bytes] = None,
        denoising_strength: float = 0.75
    ) -> Dict[str, Any]:
        """Generate image using backend"""
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_type": model_type,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "sampler": sampler,
            "batch_size": 1
        }
        
        # Add img2img if provided
        if init_image:
            payload["init_image"] = base64.b64encode(init_image).decode()
            payload["denoising_strength"] = denoising_strength
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Backend error: {error}")
                
                result = await response.json()
                
                # Decode first image
                image_b64 = result["images"][0]
                image_data = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_data))
                
                return {
                    "image": image,
                    "image_bytes": image_data,
                    "seed": result["seed"],
                    "parameters": result["parameters"],
                    "generation_time": result["generation_time"]
                }
    
    async def check_health(self) -> bool:
        """Check if backend is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def list_models(self) -> Dict[str, List[str]]:
        """List available models"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/models") as response:
                return await response.json()
    
    async def load_model(self, model_type: str) -> bool:
        """Preload a specific model"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/load_model/{model_type}",
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                return response.status == 200