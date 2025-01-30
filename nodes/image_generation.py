import torch
import numpy as np
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

class JanusImageGeneration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("JANUS_MODEL",),
                "processor": ("JANUS_PROCESSOR",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful photo of"
                }),
                "seed": ("INT", {
                    "default": 666666666666666,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "Janus-Pro"

    def generate_images(self, model, processor, prompt, seed, batch_size=1, temperature=1.0, cfg_weight=5.0, top_p=0.95):
        try:
            from janus.models import MultiModalityCausalLM
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        # 设置随机种子
        torch.manual_seed(seed)
        torch.manual_seed(seed)

        # 图像参数设置
        image_token_num = 576  # 24x24 patches
        img_size = 384  # 输出图像大小
        patch_size = 16  # 每个patch的大小
        parallel_size = batch_size

        # 准备对话格式
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 准备输入
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + processor.image_start_tag

        # 编码输入文本
        input_ids = processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        # 准备条件和无条件输入
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int, device=device)
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:  # 无条件输入
                tokens[i, 1:-1] = processor.pad_id

        # 获取文本嵌入
        inputs_embeds = model.language_model.get_input_embeddings()(tokens)
        print(f"inputs_embeds shape: {inputs_embeds.shape}")

        # 生成图像tokens
        generated_tokens = torch.zeros(
            (parallel_size * 2, image_token_num),
            dtype=torch.int,
            device=device
        )
        outputs = None
        past_key_values = None

        # 自回归生成
        for i in range(image_token_num):
            print(f"Before model call, inputs_embeds shape: {inputs_embeds.shape}")
            if i == 0:
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True
                )
            else:
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values
                )
            print(f"After model call, outputs.last_hidden_state shape: {outputs.last_hidden_state.shape}")
            hidden_states = outputs.last_hidden_state
            print(f"hidden_states shape: {hidden_states.shape}")
            past_key_values = outputs.past_key_values

            # 获取logits并应用CFG
            logits = model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[0::2, i] = next_token.squeeze(-1)  # 条件分支
            generated_tokens[1::2, i] = next_token.squeeze(-1)  # 无条件分支

            # 准备下一步的输入
            next_token = next_token.repeat_interleave(2, dim=0)  # 形状 [parallel_size*2, 1]
            img_embeds = model.prepare_gen_img_embeds(next_token)

            print(f"img_embeds shape: {img_embeds.shape}")
            # 修正维度处理逻辑
            if img_embeds.dim() == 3 and img_embeds.size(1) == 1:
                img_embeds = img_embeds.squeeze(1)  # 从 [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
            elif img_embeds.dim() != 2:
                raise ValueError(f"Unexpected img_embeds shape: {img_embeds.shape}")

            inputs_embeds = torch.cat([inputs_embeds, img_embeds.unsqueeze(1)], dim=1)
            print(f"Updated inputs_embeds shape: {inputs_embeds.shape}")

            # 在拼接前添加断言验证维度
            assert inputs_embeds.dim() == 3, f"inputs_embeds should be 3D, got {inputs_embeds.dim()}"
            assert img_embeds.unsqueeze(1).dim() == 3, f"new embeds should be 3D, got {img_embeds.unsqueeze(1).dim()}"

        # 解码生成的tokens为图像
        dec = model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )

        # 转换为numpy进行处理
        dec = dec.to(torch.float32).cpu().numpy()

        # 确保是BCHW格式
        if dec.shape[1] != 3:
            dec = np.repeat(dec, 3, axis=1)

        # 从[-1,1]转换到[0,1]
        dec = (dec + 1) / 2

        # 确保值范围在[0,1]之间
        dec = np.clip(dec, 0, 1)

        # 转换为ComfyUI需要的格式 [B,C,H,W] -> [B,H,W,C]
        dec = np.transpose(dec, (0, 2, 3, 1))

        # 转换为tensor
        images = torch.from_numpy(dec).float()

        # 确保格式正确
        assert images.ndim == 4 and images.shape[-1] == 3, f"Unexpected shape: {images.shape}"

        return (images,)

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed