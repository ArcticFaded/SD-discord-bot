"""
FastAPI backend for DiffSynth-Engine image generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import io
import base64
from PIL import Image
import torch
import uuid
from datetime import datetime
from pathlib import Path
import json
from contextlib import asynccontextmanager

# DiffSynth imports
from diffsynth import (
    ModelManager,
    SDImagePipeline,
    SDXLImagePipeline,
    SD3ImagePipeline,
    FluxImagePipeline,
    download_models
)

class GenerationRequest(BaseModel):
    """Request model for image generation"""
    prompt: str
    negative_prompt: str = ""
    model_type: str = "sd"  # sd, sdxl, sd3, flux
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.5
    seed: int = -1
    sampler: str = "euler_a"
    batch_size: int = 1
    
    # Optional img2img
    init_image: Optional[str] = None  # base64 encoded
    denoising_strength: float = 0.75
    
    # Optional controlnet
    control_image: Optional[str] = None  # base64 encoded
    control_strength: float = 1.0

class GenerationResponse(BaseModel):
    """Response model for image generation"""
    images: List[str]  # base64 encoded images
    seed: int
    generation_time: float
    parameters: Dict[str, Any]

class ModelConfig:
    """Configuration for available models"""
    def __init__(self):
        self.models = {
            "sd": {
                "model_id": "stabilityai/stable-diffusion-v1-5",
                "pipeline_class": SDImagePipeline,
                "default_size": (512, 512)
            },
            "sdxl": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "pipeline_class": SDXLImagePipeline,
                "default_size": (1024, 1024)
            },
            "sd3": {
                "model_id": "stabilityai/stable-diffusion-3-medium",
                "pipeline_class": SD3ImagePipeline,
                "default_size": (1024, 1024)
            },
            "flux": {
                "model_id": "black-forest-labs/FLUX.1-schnell",
                "pipeline_class": FluxImagePipeline,
                "default_size": (1024, 1024)
            }
        }
        self.loaded_models: Dict[str, Any] = {}
        self.model_manager = ModelManager()

class DiffSynthBackend:
    """Backend manager for DiffSynth-Engine"""
    
    def __init__(self):
        self.config = ModelConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_pipeline = None
        
    async def initialize(self):
        """Initialize backend and download default models"""
        print(f"🚀 Initializing DiffSynth backend on {self.device}")
        
        # Download default model if needed
        default_model = "sd"
        if default_model not in self.config.loaded_models:
            await self.load_model(default_model)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.current_pipeline:
            del self.current_pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    async def load_model(self, model_type: str):
        """Load a specific model"""
        if model_type not in self.config.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model_type == self.current_model:
            return  # Already loaded
        
        print(f"Loading model: {model_type}")
        
        # Clear previous model
        if self.current_pipeline:
            del self.current_pipeline
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load new model
        model_config = self.config.models[model_type]
        model_id = model_config["model_id"]
        
        # Download if needed
        download_models([model_id])
        
        # Load model
        models = self.config.model_manager.load_models([model_id])
        
        # Create pipeline
        pipeline_class = model_config["pipeline_class"]
        self.current_pipeline = pipeline_class.from_model_manager(
            self.config.model_manager
        )
        
        self.current_model = model_type
        self.config.loaded_models[model_type] = True
        
        print(f"✅ Model {model_type} loaded successfully")
    
    async def generate_image(self, request: GenerationRequest) -> GenerationResponse:
        """Generate image using DiffSynth"""
        start_time = datetime.now()
        
        # Load model if needed
        await self.load_model(request.model_type)
        
        # Set seed
        if request.seed == -1:
            request.seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(request.seed)
        
        # Prepare parameters
        gen_params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": request.steps,
            "guidance_scale": request.cfg_scale,
            "num_images_per_prompt": request.batch_size,
        }
        
        # Handle img2img
        if request.init_image:
            init_image_data = base64.b64decode(request.init_image)
            init_image = Image.open(io.BytesIO(init_image_data))
            gen_params["image"] = init_image
            gen_params["strength"] = request.denoising_strength
        
        # Generate
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    result = self.current_pipeline(**gen_params)
            else:
                result = self.current_pipeline(**gen_params)
        
        # Convert images to base64
        images_b64 = []
        for img in result.images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            images_b64.append(img_b64)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return GenerationResponse(
            images=images_b64,
            seed=request.seed,
            generation_time=generation_time,
            parameters=gen_params
        )

# Global backend instance
backend = DiffSynthBackend()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage backend lifecycle"""
    await backend.initialize()
    yield
    await backend.cleanup()

# Create FastAPI app
app = FastAPI(
    title="DiffSynth Image Generation API",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate image endpoint"""
    try:
        result = await backend.generate_image(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": list(backend.config.models.keys()),
        "loaded": list(backend.config.loaded_models.keys()),
        "current": backend.current_model
    }

@app.post("/load_model/{model_type}")
async def load_model(model_type: str):
    """Preload a specific model"""
    try:
        await backend.load_model(model_type)
        return {"status": "success", "model": model_type}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": backend.device,
        "current_model": backend.current_model,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)