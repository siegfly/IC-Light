# IC-Light

IC-Light is a project to manipulate the illumination of images.

The name "IC-Light" stands for **"Imposing Consistent Light"** (we will briefly describe this at the end of this page).

Currently, we release two types of models: text-conditioned relighting model and background-conditioned model. Both types take foreground images as inputs.

**Note that "iclightai dot com" is a scam website. They have no relationship with us. Do not give scam websites money! This GitHub repo is the only official IC-Light.**

# News

[Alternative model](https://github.com/lllyasviel/IC-Light/discussions/109) for stronger illumination modifications.

Some news about flux is [here](https://github.com/lllyasviel/IC-Light/discussions/98). (A fix [update](https://github.com/lllyasviel/IC-Light/discussions/98#discussioncomment-11370266) is added at Nov 25, more demos will be uploaded soon.)

# Get Started

Below script will run the text-conditioned relighting model:

    git clone https://github.com/lllyasviel/IC-Light.git
    cd IC-Light
    conda create -n iclight python=3.10
    conda activate iclight
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python gradio_demo.py

Or, to use background-conditioned demo:

    python gradio_demo_bg.py

Model downloading is automatic.

Note that the "gradio_demo.py" has an official [huggingFace Space here](https://huggingface.co/spaces/lllyasviel/IC-Light).

# Screenshot

### Text-Conditioned Model

(Note that the "Lighting Preference" are just initial latents - eg., if the Lighting Preference is "Left" then initial latent is left white right black.)

---

**Prompt: beautiful woman, detailed face, warm atmosphere, at home, bedroom**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/87265483-aa26-4d2e-897d-b58892f5fdd7)

---

**Prompt: beautiful woman, detailed face, sunshine from window**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/148c4a6d-82e7-4e3a-bf44-5c9a24538afc)

---

**beautiful woman, detailed face, neon, Wong Kar-wai, warm**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/f53c9de2-534a-42f4-8272-6d16a021fc01)

---

**Prompt: beautiful woman, detailed face, sunshine, outdoor, warm atmosphere**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/25d6ea24-a736-4a0b-b42d-700fe8b2101e)

---

**Prompt: beautiful woman, detailed face, sunshine, outdoor, warm atmosphere**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/dd30387b-0490-46ee-b688-2191fb752e68)

---

**Prompt: beautiful woman, detailed face, sunshine from window**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/6c9511ca-f97f-401a-85f3-92b4442000e3)

---

**Prompt: beautiful woman, detailed face, shadow from window**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/e73701d5-890e-4b15-91ee-97f16ea3c450)

---

**Prompt: beautiful woman, detailed face, sunset over sea**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/ff26ac3d-1b12-4447-b51f-73f7a5122a05)

---

**Prompt: handsome boy, detailed face, neon light, city**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/d7795e02-46f7-444f-93e7-4d6460840437)

---

**Prompt: beautiful woman, detailed face, light and shadow**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/706f70a8-d1a0-4e0b-b3ac-804e8e231c0f)

(beautiful woman, detailed face, soft studio lighting)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fe0a72df-69d4-4e11-b661-fb8b84d0274d)

---

**Prompt: Buddha, detailed face, sci-fi RGB glowing, cyberpunk**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/68d60c68-ce23-4902-939e-11629ccaf39a)

---

**Prompt: Buddha, detailed face, natural lighting**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/1841d23d-0a0d-420b-a5ab-302da9c47c17)

---

**Prompt: toy, detailed face, shadow from window**

Lighting Preference: Bottom

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/dcb97439-ea6b-483e-8e68-cf5d320368c7)

---

**Prompt: toy, detailed face, sunset over sea**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/4f78b897-621d-4527-afa7-78d62c576100)

---

**Prompt: dog, magic lit, sci-fi RGB glowing, studio lighting**

Lighting Preference: Bottom

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/1db9cac9-8d3f-4f40-82e2-e3b0cafd8613)

---

**Prompt: mysteriou human, warm atmosphere, warm atmosphere, at home, bedroom**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/5d5aa7e5-8cbd-4e1f-9f27-2ecc3c30563a)

---

### Background-Conditioned Model

The background conditioned model does not require careful prompting. One can just use simple prompts like "handsome man, cinematic lighting".

---

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/0b2a889f-682b-4393-b1ec-2cabaa182010)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/477ca348-bd47-46ff-81e6-0ffc3d05feb2)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/5bc9d8d9-02cd-442e-a75c-193f115f2ad8)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/a35e4c57-e199-40e2-893b-cb1c549612a9)

---

A more structured visualization:

![r1](https://github.com/lllyasviel/IC-Light/assets/19834515/c1daafb5-ac8b-461c-bff2-899e4c671ba3)

# Imposing Consistent Light

In HDR space, illumination has a property that all light transports are independent. 

As a result, the blending of appearances of different light sources is equivalent to the appearance with mixed light sources:

![cons](https://github.com/lllyasviel/IC-Light/assets/19834515/27c67787-998e-469f-862f-047344e100cd)

Using the above [light stage](https://www.pauldebevec.com/Research/LS/) as an example, the two images from the "appearance mixture" and "light source mixture" are consistent (mathematically equivalent in HDR space, ideally).

We imposed such consistency (using MLPs in latent space) when training the relighting models.

As a result, the model is able to produce highly consistent relight - **so** consistent that different relightings can even be merged as normal maps! Despite the fact that the models are latent diffusion.

![r2](https://github.com/lllyasviel/IC-Light/assets/19834515/25068f6a-f945-4929-a3d6-e8a152472223)

From left to right are inputs, model outputs relighting, devided shadow image, and merged normal maps. Note that the model is not trained with any normal map data. This normal estimation comes from the consistency of relighting.

You can reproduce this experiment using this button (it is 4x slower because it relight image 4 times)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/d9c37bf7-2136-446c-a9a5-5a341e4906de)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fcf5dd55-0309-4e8e-9721-d55931ea77f0)

Below are bigger images (feel free to try yourself to get more results!)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/12335218-186b-4c61-b43a-79aea9df8b21)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/2daab276-fdfa-4b0c-abcb-e591f575598a)

For reference, [geowizard](https://fuxiao0719.github.io/projects/geowizard/) (geowizard is a really great work!):

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/4ba1a96d-e218-42ab-83ae-a7918d56ee5f)

And, [switchlight](https://arxiv.org/pdf/2402.18848) (switchlight is another great work!):

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fbdd961f-0b26-45d2-802e-ffd734affab8)

# Model Notes

* **iclight_sd15_fc.safetensors** - The default relighting model, conditioned on text and foreground. You can use initial latent to influence the relighting.

* **iclight_sd15_fcon.safetensors** - Same as "iclight_sd15_fc.safetensors" but trained with offset noise. Note that the default "iclight_sd15_fc.safetensors" outperform this model slightly in a user study. And this is the reason why the default model is the model without offset noise.

* **iclight_sd15_fbc.safetensors** - Relighting model conditioned with text, foreground, and background.

Also, note that the original [BRIA RMBG 1.4](https://huggingface.co/briaai/RMBG-1.4) is for non-commercial use. If you use IC-Light in commercial projects, replace it with other background replacer like [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).

# Cite

    @inproceedings{
        zhang2025scaling,
        title={Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing Consistent Light Transport},
        author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=u1cQYxRI1H}
    }

# Related Work

Also read ...

[Total Relighting: Learning to Relight Portraits for Background Replacement](https://augmentedperception.github.io/total_relighting/)

[Relightful Harmonization: Lighting-aware Portrait Background Replacement](https://arxiv.org/abs/2312.06886)

[SwitchLight: Co-design of Physics-driven Architecture and Pre-training Framework for Human Portrait Relighting](https://arxiv.org/pdf/2402.18848)

# 前后景图片处理流程详解

## 1. 模型初始化阶段

### 核心模型加载
- **Stable Diffusion 基础模型**: `stablediffusionapi/realistic-vision-v51`
  - `CLIPTokenizer`: 文本分词器
  - `CLIPTextModel`: 文本编码器
  - `AutoencoderKL`: VAE编码解码器
  - `UNet2DConditionModel`: 扩散模型核心
- **背景移除模型**: `BriaRMBG` (`briaai/RMBG-1.4`)
- **IC-Light 专用权重**: `iclight_sd15_fbc.safetensors`

### UNet 架构修改
- 将输入通道从4个扩展到12个（原始4个 + 前景4个 + 背景4个）
- 加载IC-Light权重并与原始权重合并
- 修改前向传播函数以支持条件拼接

## 2. 图片预处理阶段

### 前景图片处理 (`run_rmbg` 函数)
1. **尺寸调整**: 使用 `resize_without_crop` 调整到合适尺寸
2. **格式转换**: `numpy2pytorch` 转换为PyTorch张量
3. **背景移除**: 使用 `BriaRMBG` 模型生成alpha遮罩
4. **插值处理**: 使用双线性插值将alpha调整到原始尺寸
5. **主体变换**（可选）:
   - 缩放调整 (`subject_scale`)
   - 位置偏移 (`subject_x_offset`, `subject_y_offset`)
6. **前景合成**: 应用alpha遮罩和sigma噪声

### 背景图片处理
根据 `bg_source` 参数选择背景来源：
- **UPLOAD**: 直接使用上传的背景图
- **UPLOAD_FLIP**: 水平翻转背景图
- **GREY/LEFT/RIGHT/TOP/BOTTOM**: 生成渐变背景

## 3. 核心处理流程 (`process` 函数)

### 第一阶段：低分辨率生成
1. **图片预处理**:
   - `resize_and_center_crop`: 调整前景和背景到目标尺寸
   - `numpy2pytorch`: 转换为PyTorch格式

2. **VAE编码**:
   - 使用 `vae.encode()` 将图片编码到潜在空间
   - 应用缩放因子 `vae.config.scaling_factor`
   - 拼接前景和背景条件 `concat_conds`

3. **文本编码**:
   - `encode_prompt_pair`: 处理正负提示词
   - `encode_prompt_inner`: 分块处理长文本
   - 使用 `CLIPTextModel` 生成文本嵌入

4. **扩散生成**:
   - 使用 `StableDiffusionPipeline` (t2i_pipe)
   - 传入条件：`concat_conds`（图片条件）和文本嵌入
   - 输出潜在空间表示

5. **VAE解码**:
   - `vae.decode()` 将潜在表示转换为图片
   - `pytorch2numpy` 转换为numpy格式

### 第二阶段：高分辨率优化
1. **分辨率提升**:
   - 根据 `highres_scale` 计算新尺寸
   - `resize_without_crop` 调整图片尺寸

2. **重新编码**:
   - 将高分辨率图片重新编码到潜在空间
   - 重新计算前景背景条件

3. **图生图优化**:
   - 使用 `StableDiffusionImg2ImgPipeline` (i2i_pipe)
   - 应用 `highres_denoise` 强度进行去噪
   - 步数调整为 `steps / highres_denoise`

4. **最终输出**:
   - VAE解码得到最终图片
   - 返回处理结果和额外图片

## 4. 特殊处理模式

### 重光照模式 (`process_relight`)
- 使用 `sigma=0.0` 进行背景移除
- 调用标准 `process` 函数
- 结果量化到0-255范围

### 法线贴图模式 (`process_normal`)
- 使用 `sigma=16` 进行背景移除
- 分别生成4个方向的光照（左、右、上、下）
- 计算环境光照和方向向量
- 生成法线贴图和各方向光照结果

## 5. 关键技术特点

### 条件控制机制
- **前景条件**: 经过背景移除的主体图像
- **背景条件**: 目标背景或生成的渐变背景
- **文本条件**: 用户输入的描述性提示词

### 多尺度处理
- 先在较低分辨率生成基础结果
- 再通过图生图管道进行高分辨率优化
- 保证生成质量的同时提高效率

### 设备优化
- 模型加载到CUDA设备
- 使用混合精度（float16/bfloat16）
- 应用 `AttnProcessor2_0` 优化注意力计算

这个处理流程实现了高质量的前景主体重光照，能够根据指定背景和文本描述生成逼真的光照效果。
