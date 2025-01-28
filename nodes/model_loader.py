import os
import shutil
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from janus.models import VLChatProcessor
from safetensors.torch import load_file, save_file
import torch


class JanusModelLoader:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"],),
            },
        }
    
    RETURN_TYPES = ("JANUS_MODEL", "JANUS_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "Janus-Pro"

    def find_weight_path(self, model_dir):
        """Find model weight file with improved shard handling"""
        # First check for standard files
        standard_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(standard_path):
            return standard_path
            
        # Check for sharded formats - return first shard, transformers will handle the rest
        safetensors_shards = sorted([
            f for f in os.listdir(model_dir) 
            if f.startswith("model-") and f.endswith(".safetensors")
        ])
        if safetensors_shards:
            return os.path.join(model_dir, safetensors_shards[0])
            
        # Fallback to .bin files if they exist
        pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(pytorch_path):
            return pytorch_path
            
        pytorch_shards = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith("pytorch_model-") and f.endswith(".bin")
        ])
        if pytorch_shards:
            return os.path.join(model_dir, pytorch_shards[0])
            
        return None

    def load_model(self, model_name):
        try:
            import torch
        except ImportError:
            raise ImportError("Please install PyTorch using 'pip install torch'.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16

        # Get ComfyUI root directory and model path
        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        model_dir = os.path.join(comfy_path, "models", "Janus-Pro", os.path.basename(model_name))
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # First download/verify support files
            config = AutoConfig.from_pretrained(model_name)
            config.save_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_dir)
            processor = VLChatProcessor.from_pretrained(model_name)
            processor.save_pretrained(model_dir)
            
            # Check for model weights
            weight_path = self.find_weight_path(model_dir)
            if not weight_path:
                print("Downloading model weights...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    device_map="auto"
                )
                # Save with default configuration and safetensors format
                model.save_pretrained(
                    model_dir,
                    safe_serialization=True
                )
                
                # Find the weight path again after download
                weight_path = self.find_weight_path(model_dir)
                if not weight_path:
                    raise FileNotFoundError("Model weights were not found after download")
                    
            # Load the validated model and processor
            vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto"
            )
            
            return (vl_gpt, vl_chat_processor)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load/validate model: {str(e)}")
