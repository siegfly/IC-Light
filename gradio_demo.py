import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples
import re
import requests
import json

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file

# 翻译功能
def contains_chinese(text):
    """检测文本是否包含中文字符"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def translate_to_english(text):
    """将中文文本翻译为英文，如果是英文则不变"""
    if not text or not text.strip():
        return text
    
    # 如果不包含中文，直接返回原文
    if not contains_chinese(text):
        return text
    
    try:
        # 使用免费的翻译API (Google Translate)
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'zh',  # 源语言：中文
            'tl': 'en',  # 目标语言：英文
            'dt': 't',
            'q': text
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and len(result[0]) > 0:
                translated_text = ''.join([item[0] for item in result[0] if item[0]])
                return translated_text
    except Exception as e:
        print(f"翻译失败: {e}")
        # 如果翻译失败，返回原文
        return text
    
    return text


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = './models/iclight_sd15_fc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0, subject_scale=1.0, subject_x_offset=0.0, subject_y_offset=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    
    # 应用主体缩放和位置调整
    if subject_scale != 1.0 or subject_x_offset != 0.0 or subject_y_offset != 0.0:
        # 确保alpha是2D数组
        alpha = alpha.squeeze() if len(alpha.shape) > 2 else alpha
        # 创建变换后的前景和alpha
        transformed_img = np.zeros_like(img)
        transformed_alpha = np.zeros((H, W), dtype=alpha.dtype)
        
        # 计算缩放后的尺寸
        scaled_h = int(H * subject_scale)
        scaled_w = int(W * subject_scale)
        
        # 缩放图像和alpha
        if scaled_h > 0 and scaled_w > 0:
            scaled_img = resize_without_crop(img, scaled_w, scaled_h)
            # 确保alpha是2D数组
            alpha_2d = alpha if len(alpha.shape) == 2 else alpha.squeeze()
            # 将alpha转换为3通道图像进行缩放，然后取第一个通道
            alpha_3ch = np.stack([alpha_2d, alpha_2d, alpha_2d], axis=-1)
            scaled_alpha_3ch = resize_without_crop((alpha_3ch * 255).astype(np.uint8), scaled_w, scaled_h)
            scaled_alpha = (scaled_alpha_3ch[:, :, 0].astype(np.float32)) / 255.0
            
            # 计算放置位置（考虑偏移）
            center_x = W // 2 + int(subject_x_offset * W)
            center_y = H // 2 + int(subject_y_offset * H)
            
            start_x = max(0, center_x - scaled_w // 2)
            start_y = max(0, center_y - scaled_h // 2)
            end_x = min(W, start_x + scaled_w)
            end_y = min(H, start_y + scaled_h)
            
            # 计算在缩放图像中的对应区域
            src_start_x = max(0, scaled_w // 2 - center_x) if center_x < scaled_w // 2 else 0
            src_start_y = max(0, scaled_h // 2 - center_y) if center_y < scaled_h // 2 else 0
            
            # 确保目标区域和源区域尺寸匹配
            dst_h = end_y - start_y
            dst_w = end_x - start_x
            src_end_x = min(scaled_w, src_start_x + dst_w)
            src_end_y = min(scaled_h, src_start_y + dst_h)
            
            # 重新调整目标区域以匹配实际可用的源区域
            actual_w = src_end_x - src_start_x
            actual_h = src_end_y - src_start_y
            end_x = start_x + actual_w
            end_y = start_y + actual_h
            
            # 放置缩放后的图像和alpha
            if end_x > start_x and end_y > start_y and src_end_x > src_start_x and src_end_y > src_start_y:
                transformed_img[start_y:end_y, start_x:end_x] = scaled_img[src_start_y:src_end_y, src_start_x:src_end_x]
                # scaled_alpha现在确保是2D数组
                transformed_alpha[start_y:end_y, start_x:end_x] = scaled_alpha[src_start_y:src_end_y, src_start_x:src_end_x]
        
        # 使用变换后的图像和alpha
        img = transformed_img
        alpha = transformed_alpha
    
    # 确保alpha是2D数组
    if len(alpha.shape) > 2:
        alpha = alpha.squeeze()
    if len(alpha.shape) > 2:
        alpha = alpha[:, :, 0] if alpha.shape[2] == 1 else alpha.mean(axis=2)
    
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha[..., np.newaxis]
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, subject_scale, subject_x_offset, subject_y_offset):
    input_fg, matting = run_rmbg(input_fg, sigma=0.0, subject_scale=subject_scale, subject_x_offset=subject_x_offset, subject_y_offset=subject_y_offset)
    results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    return input_fg, results


quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm'
]
quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground Condition)")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(label="Image", height=480, type="numpy")
                output_bg = gr.Image(label="Preprocessed Foreground", height=480, type="numpy")
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", scale=4)
                translate_prompt_btn = gr.Button("🌐 翻译", size="sm", scale=1)
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.NONE.value,
                                 label="Lighting Preference (Initial Latent)", type='value')
            example_quick_subjects = gr.Dataset(samples=quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)

                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

            with gr.Accordion("Advanced options", open=False):
                with gr.Row():
                    subject_scale = gr.Slider(label="Subject Scale", minimum=0.1, maximum=3.0, value=1.0, step=0.01)
                with gr.Row():
                    subject_x_offset = gr.Slider(label="Subject X Offset", minimum=-0.5, maximum=0.5, value=0.0, step=0.01)
                    subject_y_offset = gr.Slider(label="Subject Y Offset", minimum=-0.5, maximum=0.5, value=0.0, step=0.01)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2, step=0.01)
                lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                with gr.Row():
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality', scale=4)
                    translate_a_prompt_btn = gr.Button("🌐 翻译", size="sm", scale=1)
                with gr.Row():
                    n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality', scale=4)
                    translate_n_prompt_btn = gr.Button("🌐 翻译", size="sm", scale=1)
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')
    with gr.Row():
        dummy_image_for_outputs = gr.Image(visible=False, label='Result')
        gr.Examples(
            fn=lambda *args: ([args[-1]], None),
            examples=db_examples.foreground_conditioned_examples,
            inputs=[
                input_fg, prompt, bg_source, image_width, image_height, seed, dummy_image_for_outputs
            ],
            outputs=[result_gallery, output_bg],
            run_on_click=True, examples_per_page=1024
        )
    ips = [input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, subject_scale, subject_x_offset, subject_y_offset]
    relight_button.click(fn=process_relight, inputs=ips, outputs=[output_bg, result_gallery])
    example_quick_prompts.click(lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), inputs=[example_quick_prompts, prompt], outputs=prompt, show_progress=False, queue=False)
    example_quick_subjects.click(lambda x: x[0], inputs=example_quick_subjects, outputs=prompt, show_progress=False, queue=False)
    
    # 翻译按钮事件
    translate_prompt_btn.click(fn=translate_to_english, inputs=prompt, outputs=prompt, show_progress=False, queue=False)
    translate_a_prompt_btn.click(fn=translate_to_english, inputs=a_prompt, outputs=a_prompt, show_progress=False, queue=False)
    translate_n_prompt_btn.click(fn=translate_to_english, inputs=n_prompt, outputs=n_prompt, show_progress=False, queue=False)


block.launch(server_name='127.0.0.1')
