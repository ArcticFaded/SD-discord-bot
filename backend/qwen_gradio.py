#!/usr/bin/env python3
"""
Gradio interface for Qwen-Image model testing
Supports text-to-image and image+text-to-image (img2img)
"""

import gradio as gr
import torch
from PIL import Image
import numpy as np
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import time
import random
import os

class QwenImageGradioApp:
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the Qwen-Image model"""
        if self.is_loaded:
            return "Model already loaded!"
        
        try:
            print("Loading Qwen-Image model...")
            self.pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=self.torch_dtype,
                device=self.device,
                model_configs=[
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
                ],
                tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
            )
            
            # Optional: Enable VRAM management for lower memory usage
            if torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                if available_vram < 24:  # If less than 24GB VRAM
                    print(f"Enabling VRAM management (detected {available_vram:.1f}GB)")
                    self.pipe.enable_vram_management(vram_buffer=2.0)
            
            self.is_loaded = True
            print("Model loaded successfully!")
            return "✅ Model loaded successfully!"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def generate_image(
        self,
        prompt,
        negative_prompt,
        input_image,
        denoising_strength,
        width,
        height,
        num_inference_steps,
        cfg_scale,
        seed,
        use_random_seed,
        tiled,
        tile_size,
        tile_stride,
        progress=gr.Progress()
    ):
        """Generate image with Qwen-Image model"""
        
        if not self.is_loaded:
            yield None, "Please load the model first!"
            return
        
        try:
            # Handle seed
            if use_random_seed:
                seed = random.randint(0, 2**32 - 1)
            
            # Progress callback
            progress(0, desc="Starting generation...")
            
            # Prepare parameters
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "rand_device": "cpu",
                "tiled": tiled,
                "tile_size": tile_size,
                "tile_stride": tile_stride,
            }
            
            # Add img2img parameters if input image provided
            if input_image is not None:
                gen_kwargs["input_image"] = input_image
                gen_kwargs["denoising_strength"] = denoising_strength
                progress(0.1, desc="Processing input image...")
            
            # Generate
            progress(0.2, desc="Generating image...")
            start_time = time.time()
            
            # Custom progress bar that updates Gradio
            class GradioProgress:
                def __init__(self, gr_progress, total_steps):
                    self.gr_progress = gr_progress
                    self.total_steps = total_steps
                    self.current = 0
                
                def __iter__(self):
                    return self
                
                def __next__(self):
                    if self.current < len(self.items):
                        item = self.items[self.current]
                        self.current += 1
                        self.gr_progress(
                            0.2 + (0.7 * self.current / self.total_steps),
                            desc=f"Step {self.current}/{self.total_steps}"
                        )
                        return item
                    raise StopIteration
                
                def __call__(self, items):
                    self.items = list(items)
                    self.total_steps = len(self.items)
                    self.current = 0
                    return self
            
            gen_kwargs["progress_bar_cmd"] = GradioProgress(progress, num_inference_steps)
            
            # Generate image
            image = self.pipe(**gen_kwargs)
            
            generation_time = time.time() - start_time
            progress(1.0, desc="Generation complete!")
            
            # Prepare info
            info = f"""
✅ Generation Complete!
⏱️ Time: {generation_time:.2f}s
🎲 Seed: {seed}
📐 Size: {width}x{height}
🔧 Steps: {num_inference_steps}
⚡ CFG Scale: {cfg_scale}
{"🖼️ Mode: Image-to-Image" if input_image else "✏️ Mode: Text-to-Image"}
{"🔸 Denoising: " + str(denoising_strength) if input_image else ""}
{"🏗️ Tiled: Yes" if tiled else ""}
            """
            
            yield image, info.strip()
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            print(error_msg)
            yield None, error_msg

def create_interface():
    """Create Gradio interface"""
    app = QwenImageGradioApp()
    
    with gr.Blocks(title="Qwen-Image Model Tester", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 Qwen-Image Model Tester
        
        Test the Qwen-Image diffusion model with text-to-image and image-to-image capabilities.
        
        ### Features:
        - 🖼️ Text-to-Image generation
        - 🎨 Image-to-Image transformation (img2img)
        - 🏗️ Tiled generation for high-resolution images
        - 💾 VRAM management for GPUs with limited memory
        """)
        
        with gr.Row():
            with gr.Column():
                load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
                load_status = gr.Textbox(label="Status", value="Model not loaded", interactive=False)
        
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Text inputs
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        value="精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",
                        lines=3
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid...",
                        value="",
                        lines=2
                    )
                    
                    # Image input for img2img
                    with gr.Accordion("🎨 Image-to-Image (Optional)", open=False):
                        input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=256
                        )
                        denoising_strength = gr.Slider(
                            label="Denoising Strength",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            info="Lower = more similar to input, Higher = more creative"
                        )
                    
                    # Generation parameters
                    with gr.Accordion("⚙️ Generation Settings", open=True):
                        with gr.Row():
                            width = gr.Slider(
                                label="Width",
                                minimum=512,
                                maximum=2048,
                                value=1024,
                                step=64
                            )
                            height = gr.Slider(
                                label="Height",
                                minimum=512,
                                maximum=2048,
                                value=1024,
                                step=64
                            )
                        
                        with gr.Row():
                            num_inference_steps = gr.Slider(
                                label="Steps",
                                minimum=1,
                                maximum=100,
                                value=30,
                                step=1
                            )
                            cfg_scale = gr.Slider(
                                label="CFG Scale",
                                minimum=1.0,
                                maximum=20.0,
                                value=4.0,
                                step=0.5
                            )
                        
                        with gr.Row():
                            seed = gr.Number(
                                label="Seed",
                                value=42,
                                precision=0
                            )
                            use_random_seed = gr.Checkbox(
                                label="Random Seed",
                                value=False
                            )
                    
                    # Tiling options for high-res
                    with gr.Accordion("🏗️ Tiling Options (for high-res)", open=False):
                        tiled = gr.Checkbox(
                            label="Enable Tiled Generation",
                            value=False,
                            info="Use for very high resolution images to save VRAM"
                        )
                        with gr.Row():
                            tile_size = gr.Slider(
                                label="Tile Size",
                                minimum=64,
                                maximum=256,
                                value=128,
                                step=32
                            )
                            tile_stride = gr.Slider(
                                label="Tile Stride",
                                minimum=32,
                                maximum=128,
                                value=64,
                                step=16
                            )
                    
                    generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Generated Image",
                        type="pil",
                        interactive=False
                    )
                    output_info = gr.Textbox(
                        label="Generation Info",
                        lines=10,
                        interactive=False
                    )
        
        with gr.Tab("Examples"):
            gr.Markdown("""
            ### Example Prompts
            
            **Chinese Portrait (中文肖像):**
            ```
            精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。
            ```
            
            **English Landscape:**
            ```
            Majestic mountain landscape at sunset, golden hour lighting, snow-capped peaks, 
            pristine lake reflection, pine forest, dramatic clouds, photorealistic, 8k quality
            ```
            
            **Abstract Art:**
            ```
            Abstract geometric composition, vibrant colors, flowing shapes, digital art style,
            neon accents, futuristic aesthetic, high contrast, dynamic movement
            ```
            
            **Character Design:**
            ```
            Anime character portrait, detailed eyes, flowing hair, magical aura, 
            fantasy clothing, soft lighting, professional illustration, high detail
            ```
            """)
        
        with gr.Tab("Settings"):
            gr.Markdown("""
            ### Model Settings
            
            - **Device**: CUDA (if available) or CPU
            - **Precision**: bfloat16 for optimal performance
            - **VRAM Management**: Automatically enabled for GPUs < 24GB
            
            ### Tips:
            
            1. **For Text-to-Image**: Leave the input image empty
            2. **For Image-to-Image**: Upload an image and adjust denoising strength
            3. **For High-Res**: Enable tiling to generate larger images with less VRAM
            4. **CFG Scale**: Lower values (3-5) for more creative, higher (7-15) for prompt adherence
            """)
        
        # Event handlers
        load_btn.click(
            fn=app.load_model,
            outputs=[load_status]
        )
        
        generate_btn.click(
            fn=app.generate_image,
            inputs=[
                prompt,
                negative_prompt,
                input_image,
                denoising_strength,
                width,
                height,
                num_inference_steps,
                cfg_scale,
                seed,
                use_random_seed,
                tiled,
                tile_size,
                tile_stride
            ],
            outputs=[output_image, output_info]
        )
    
    return demo

if __name__ == "__main__":
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ CUDA not available, using CPU (will be slow)")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public URL
        inbrowser=True
    )