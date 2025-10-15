from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import io
from PIL import Image
import base64
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local AI Image Generator")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for model
pipe = None

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, bad quality, distorted, ugly"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int = -1

@app.on_event("startup")
async def load_model():
    """Load Stable Diffusion model on startup"""
    global pipe
    try:
        logger.info("Loading Stable Diffusion model...")
        
        model_id = "runwayml/stable-diffusion-v1-5"
        # For faster inference, you can use: "stabilityai/stable-diffusion-2-1"
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load the model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for faster inference
            requires_safety_checker=False
        )
        
        # Use DPM Solver for faster inference
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
            # Uncomment if you have limited VRAM:
            # pipe.enable_sequential_cpu_offload()
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Local AI Image Generator API",
        "status": "ready" if pipe is not None else "loading",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate")
async def generate_image(request: ImageRequest):
    """Generate image from text prompt"""
    
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        logger.info(f"Generating image with prompt: {request.prompt[:50]}...")
        
        # Set seed for reproducibility
        generator = None
        if request.seed != -1:
            generator = torch.Generator(device=pipe.device).manual_seed(request.seed)
        
        # Generate image
        with torch.inference_mode():
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                generator=generator
            )
        
        image = result.images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info("Image generated successfully!")
        
        return {
            "image": f"data:image/png;base64,{img_str}",
            "prompt": request.prompt,
            "seed": request.seed if request.seed != -1 else "random"
        }
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.post("/generate-stream")
async def generate_image_stream(request: ImageRequest):
    """Generate and stream image (useful for larger images)"""
    
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        generator = None
        if request.seed != -1:
            generator = torch.Generator(device=pipe.device).manual_seed(request.seed)
        
        with torch.inference_mode():
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                generator=generator
            )
        
        image = result.images[0]
        
        # Convert to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        return StreamingResponse(buffered, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)