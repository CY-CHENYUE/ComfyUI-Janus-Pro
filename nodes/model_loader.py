import os
import sys
import logging
import torch
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import comfy.model_management as mm
import folder_paths

class JanusModelLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"],),
            },
            "optional": {
                "force_download": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("JANUS_MODEL", "JANUS_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "Janus-Pro"

    def download_model(self, model_name, model_dir, force=False):
        """Download model from Hugging Face Hub."""
        try:
            if os.path.exists(model_dir) and not force:
                if self.check_model_files(model_dir):
                    self.logger.info(f"Model already exists at {model_dir}")
                    return True
            
            self.logger.info(f"Downloading {model_name} to {model_dir}...")
            self.logger.info("This may take a while depending on your internet speed...")
            
            # Create the directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Download with Hugging Face Hub
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=os.getenv('HF_TOKEN')
            )
            
            self.logger.info("Download completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Error downloading model: {str(e)}")
            if "401" in str(e):
                self.logger.error("Authentication error. This might be a private model requiring authentication.")
                self.logger.error("Please set your HF_TOKEN environment variable with a valid Hugging Face token.")
            elif "404" in str(e):
                self.logger.error(f"Model {model_name} not found on Hugging Face Hub.")
                self.logger.error("Please check the model name and try again.")
            return False

    def check_model_files(self, model_dir):
        """Check if all required model files are present."""
        if not os.path.exists(model_dir):
            return False
            
        required_files = ['config.json', 'tokenizer.model']
        existing_files = os.listdir(model_dir)
        
        # Check for either safetensors or pytorch format
        has_weights = any(f.endswith(('.safetensors', 'pytorch_model.bin')) 
                         for f in existing_files)
        
        if not has_weights:
            return False
            
        for req_file in required_files:
            if not any(f == req_file for f in existing_files):
                return False
                
        return True

    def load_model(self, model_name, force_download=False):
        """Load the Janus model with auto-download capability."""
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        device = mm.get_torch_device()
        
        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16

        # Get models directory from ComfyUI paths
        model_dir = os.path.join(folder_paths.models_dir, "Janus-Pro", 
                               os.path.basename(model_name))

        # Check if model needs to be downloaded
        if force_download or not self.check_model_files(model_dir):
            self.logger.info(f"Model files not found in {model_dir} or force download requested")
            success = self.download_model(model_name, model_dir, force=force_download)
            if not success:
                raise ValueError(f"Failed to download model {model_name}. Please check the logs for details.")

        try:
            self.logger.info("Loading processor...")
            vl_chat_processor = VLChatProcessor.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            
            self.logger.info("Loading model...")
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype
            )
            
            vl_gpt = vl_gpt.eval()
            
            self.logger.info("Model loaded successfully!")
            return (vl_gpt, vl_chat_processor)
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
