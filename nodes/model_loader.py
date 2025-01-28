import os
import folder_paths
import logging

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

    def load_model(self, model_name):
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Avaliable device detected {device}")
        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            logging.error(f"Using torch.bfloat16 error, fallback to torch.float16")
            dtype = torch.float16
        model_folder_paths = folder_paths.get_folder_paths("Janus-Pro")
        folder_exists = False
        for folder_path in model_folder_paths:
            if os.path.exists(folder_path):
                folder_exists = True
            logging.info(f"Searching model directory {folder_path}, exist = {folder_exists}")
            # 构建模型路径
            model_dir = os.path.join(folder_path,os.path.basename(model_name))
        if not folder_exists:
            raise ValueError(f"Local model not found at {model_dir}. Please download the model and place it in the ComfyUI/models/Janus-Pro folder.")
        
        logging.info(f"Loading vl_chat_processor from {model_dir}")
        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
        
        logging.info(f"Loading vl_gpt from {model_dir}")
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        logging.info(f"Transfer vl_gpt to {device} with {dtype}")
        vl_gpt = vl_gpt.to(dtype).to(device).eval()

        return (vl_gpt, vl_chat_processor) 