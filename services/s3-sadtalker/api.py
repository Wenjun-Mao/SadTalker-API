import sys
import os
import torch
import shutil
import time
import uuid
from time import strftime
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

# --- Configuration ---
CHECKPOINT_DIR = './checkpoints'
CONFIG_DIR = './src/config'
OUTPUT_DIR = 'output'
SIZE = 256
OLD_VERSION = False # Use safetensors if available

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global models dictionary
models = {}

def load_models(device):
    """
    Load all necessary models into the global dictionary.
    """
    print(f"Loading models on {device}...")
    
    # 1. Initialize paths for both modes
    paths_crop = init_path(CHECKPOINT_DIR, CONFIG_DIR, SIZE, OLD_VERSION, preprocess='crop')
    paths_full = init_path(CHECKPOINT_DIR, CONFIG_DIR, SIZE, OLD_VERSION, preprocess='full')

    # 2. Load Preprocess Model (CropAndExtract)
    # This seems to be shared or at least compatible enough to use one instance 
    # if we assume the underlying detector is the same.
    # However, init_path returns different 'sadtalker_paths' based on preprocess.
    # CropAndExtract uses 'path_of_net_recon_model' or 'checkpoint' (safetensor).
    # Both paths_crop and paths_full should point to the same checkpoint for net_recon.
    print("Loading Preprocess Model...")
    models['preprocess'] = CropAndExtract(paths_crop, device)

    # 3. Load Audio2Coeff Models
    print("Loading Audio2Coeff (Crop)...")
    models['audio2coeff_crop'] = Audio2Coeff(paths_crop, device)
    
    print("Loading Audio2Coeff (Full)...")
    models['audio2coeff_full'] = Audio2Coeff(paths_full, device)

    # 4. Load AnimateFromCoeff Models
    print("Loading AnimateFromCoeff (Crop)...")
    models['animate_crop'] = AnimateFromCoeff(paths_crop, device)
    
    print("Loading AnimateFromCoeff (Full)...")
    models['animate_full'] = AnimateFromCoeff(paths_full, device)
    
    print("All models loaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This service requires a GPU to run.")
    
    device = "cuda"
    
    try:
        load_models(device)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e
    
    yield
    
    # Cleanup if necessary
    models.clear()
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class InferenceRequest(BaseModel):
    driven_audio: str
    source_image: str
    option: str  # "full" or "crop"

def run_inference(driven_audio: str, source_image: str, option: str, job_id: str):
    """
    Background task to run the inference.
    """
    try:
        device = "cuda"
        
        # Determine settings based on option
        if option == "full":
            preprocess_flag = "full"
            still_mode = True
            audio_to_coeff = models['audio2coeff_full']
            animate_from_coeff = models['animate_full']
        else: # crop
            preprocess_flag = "crop"
            still_mode = False
            audio_to_coeff = models['audio2coeff_crop']
            animate_from_coeff = models['animate_crop']

        preprocess_model = models['preprocess']
        
        # Create a temporary directory for this job to hold intermediate files
        # We will name it using the job_id to ensure uniqueness
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{job_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # --- Step 1: Preprocess (Crop & Extract 3DMM) ---
            first_frame_dir = os.path.join(temp_dir, 'first_frame_dir')
            os.makedirs(first_frame_dir, exist_ok=True)
            
            # Note: CropAndExtract.generate takes 'crop_or_resize' argument which corresponds to our preprocess_flag
            first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
                source_image, 
                first_frame_dir, 
                preprocess_flag, 
                source_image_flag=True, 
                pic_size=SIZE
            )
            
            if first_coeff_path is None:
                print(f"Job {job_id}: Failed to extract coefficients from source image.")
                return

            # --- Step 2: Audio to Coeff ---
            # get_data arguments: first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still
            batch = get_data(
                first_coeff_path, 
                driven_audio, 
                device, 
                ref_eyeblink_coeff_path=None, 
                still=still_mode
            )
            
            # Audio2Coeff.generate arguments: batch, coeff_save_dir, pose_style, ref_pose_coeff_path
            # pose_style defaults to 0 in inference.py
            coeff_path = audio_to_coeff.generate(batch, temp_dir, 0, None)

            # --- Step 3: Face Render ---
            # get_facerender_data arguments: coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, ...
            # inference.py default batch_size=2
            data = get_facerender_data(
                coeff_path, 
                crop_pic_path, 
                first_coeff_path, 
                driven_audio, 
                2, # batch_size
                input_yaw_list=None, 
                input_pitch_list=None, 
                input_roll_list=None, 
                expression_scale=1.0, 
                still_mode=still_mode, 
                preprocess=preprocess_flag, 
                size=SIZE
            )
            
            # AnimateFromCoeff.generate arguments: data, video_save_dir, pic_path, crop_info, enhancer, ...
            # We always use gfpgan as per requirements
            result_path = animate_from_coeff.generate(
                data, 
                temp_dir, 
                source_image, 
                crop_info, 
                enhancer='gfpgan', 
                background_enhancer=None, 
                preprocess=preprocess_flag, 
                img_size=SIZE
            )
            
            # --- Step 4: Finalize ---
            # The result is in temp_dir. We need to move it to OUTPUT_DIR with the correct name.
            # Filename: timestamp_shortHash.mp4
            timestamp = strftime("%Y_%m_%d_%H.%M.%S")
            short_hash = job_id[:8]
            final_filename = f"{timestamp}_{short_hash}.mp4"
            final_path = os.path.join(OUTPUT_DIR, final_filename)
            
            shutil.move(result_path, final_path)
            print(f"Job {job_id}: Success. Output saved to {final_path}")

        except Exception as e:
            print(f"Job {job_id}: Error during inference: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    except Exception as e:
         print(f"Job {job_id}: Critical error: {e}")


@app.post("/inference")
async def inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    # 1. Validate inputs
    if not os.path.exists(request.driven_audio):
        raise HTTPException(status_code=400, detail=f"Audio file not found: {request.driven_audio}")
    if not os.path.exists(request.source_image):
        raise HTTPException(status_code=400, detail=f"Source image not found: {request.source_image}")
    if request.option not in ["full", "crop"]:
        raise HTTPException(status_code=400, detail="Option must be 'full' or 'crop'")

    # 2. Generate Job ID
    job_id = str(uuid.uuid4())

    # 3. Add background task
    background_tasks.add_task(
        run_inference, 
        request.driven_audio, 
        request.source_image, 
        request.option, 
        job_id
    )

    # 4. Return success signal
    return "1"
