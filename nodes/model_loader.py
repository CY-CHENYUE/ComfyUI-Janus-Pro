import os

class JanusModelLoader:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"],),
                "load_in_4bit": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("JANUS_MODEL", "JANUS_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "Janus-Pro"

    def load_model(self, model_name, load_in_4bit):
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        # 检查4位量化依赖
        if load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                raise ImportError("4-bit量化需要bitsandbytes包，请使用 pip install bitsandbytes 安装")

        # 获取ComfyUI根目录
        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        model_dir = os.path.join(
            comfy_path,
            "models",
            "Janus-Pro",
            os.path.basename(model_name)
        )
        
        if not os.path.exists(model_dir):
            raise ValueError(f"Local model not found at {model_dir}. Please download the model and place it in the ComfyUI/models/Janus-Pro folder.")

        # 加载处理器时强制指定数据类型
        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
        
        # 统一计算精度设置
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        if load_in_4bit:
            # 量化配置与处理器精度同步
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,  # 使用统一计算精度
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=compute_dtype  # 显式指定模型精度
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=compute_dtype
            )
            vl_gpt = vl_gpt.to(device).eval()

        return (vl_gpt, vl_chat_processor)