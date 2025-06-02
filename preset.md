# IC-Light 参数预设配置

## 减少背景与文本对前景影响的参数组合设置

### 1. 最小影响设置（保守型）
**适用场景**：需要最大程度保持前景原貌，只进行轻微的光照调整

**核心参数**：
- `cfg`: 3.0-5.0（降低条件引导强度）
- `steps`: 15-20（减少扩散步数）
- `highres_denoise`: 0.2-0.3（最小高分辨率处理）
- `highres_scale`: 1.2-1.5（轻微放大）

**背景设置**：
- `bg_source`: "Environment"（环境光，影响最小）
- 或使用纯色背景（灰色、白色）

**文本设置**：
- `prompt`: "natural lighting"（自然光照）
- `a_prompt`: "high quality, detailed"（质量提示）
- `n_prompt`: "overexposed, dramatic lighting, strong shadows, color cast, distorted face"（避免强烈效果）

**主体变换**：
- `subject_scale`: 0.9-1.0（保持或略微缩小）
- `subject_x_offset`: 0
- `subject_y_offset`: 0

**图像尺寸**：
- 保持原始比例，避免过度裁剪

### 2. 轻度调整设置（平衡型）
**适用场景**：在保持前景特征的同时，允许适度的光照和氛围调整

**核心参数**：
- `cfg`: 5.0-7.0
- `steps`: 20-25
- `highres_denoise`: 0.3-0.5
- `highres_scale`: 1.5-2.0

**背景设置**：
- `bg_source`: "Left Light" 或 "Right Light"（定向光源）
- 或上传简单的背景图片

**文本设置**：
- `prompt`: "soft lighting, natural shadows"（柔和光照）
- `a_prompt`: "professional photography, balanced exposure"
- `n_prompt`: "harsh lighting, extreme contrast, facial distortion, unnatural colors"

**主体变换**：
- `subject_scale`: 0.8-1.1
- 允许轻微的位置调整

### 3. 选择性影响设置（定向型）
**适用场景**：只想改变特定方面（如光照方向），其他保持不变

**核心参数**：
- `cfg`: 6.0-8.0
- `steps`: 25-30
- `highres_denoise`: 0.4-0.6
- `highres_scale`: 1.5-2.5

**背景设置**：
- 根据需求选择特定光源方向
- `bg_source`: "Top Light"、"Bottom Light" 等

**文本设置**：
- 精确描述想要的效果："rim lighting from left"（左侧轮廓光）
- `a_prompt`: "subtle changes, preserve facial features"
- `n_prompt`: "dramatic changes, face morphing, style transfer, artistic effects"

### 4. 快速预览设置（效率型）
**适用场景**：快速测试效果，减少计算时间

**核心参数**：
- `cfg`: 4.0-6.0
- `steps`: 10-15（最少步数）
- `highres_denoise`: 0.2-0.4
- `highres_scale`: 1.2-1.8

**背景设置**：
- 使用预设光源，避免复杂背景

**文本设置**：
- 简单描述："good lighting"
- `n_prompt`: "bad quality, distorted"

### 5. 前景保护设置（安全型）
**适用场景**：处理重要人像，确保面部特征不变形

**核心参数**：
- `cfg`: 2.0-4.0（极低引导强度）
- `steps`: 12-18
- `highres_denoise`: 0.1-0.25（最小去噪）
- `highres_scale`: 1.0-1.3（避免过度放大）

**背景设置**：
- `bg_source`: "Environment" 或纯色

**文本设置**：
- `prompt`: "preserve original, gentle enhancement"
- `a_prompt`: "maintain facial structure, natural look"
- `n_prompt`: "face distortion, morphing, style change, dramatic lighting, strong effects"

**主体变换**：
- `subject_scale`: 0.95-1.0（保持原始大小）
- 位置偏移为0

## 通用减少影响技巧

### 文本策略
1. **使用保守性词汇**："natural"、"subtle"、"gentle"、"preserve"
2. **避免强烈描述**：避免"dramatic"、"intense"、"artistic"、"stylized"
3. **添加保护性提示**："maintain original features"、"keep facial structure"

### 负向提示词模板
```
face distortion, facial morphing, style transfer, dramatic lighting, 
overexposed, underexposed, color cast, unnatural colors, 
artistic effects, painting style, cartoon, anime, 
strong shadows, harsh lighting, extreme contrast
```

### 参数组合原则
1. **低CFG + 少步数** = 最小影响
2. **小缩放 + 低去噪** = 保持细节
3. **简单背景 + 保守文本** = 减少条件干扰
4. **原始尺寸 + 中心位置** = 避免变形

### 测试流程建议
1. 先用"最小影响设置"测试
2. 如果效果不够，逐步提高参数
3. 重点观察面部区域的变化
4. 记录最佳参数组合供后续使用

### 参数影响程度排序

**最高影响**：
- `cfg`（CFG Scale）
- `highres_denoise`（高分辨率去噪强度）
- `prompt`（主要文本提示）

**高影响**：
- `steps`（扩散步数）
- `bg_source`（背景来源）
- `subject_scale`（主体缩放）

**中等影响**：
- `highres_scale`（高分辨率缩放）
- `a_prompt`（附加提示）
- `subject_x_offset`、`subject_y_offset`（位置偏移）

**较低影响**：
- `n_prompt`（负面提示）
- 图像尺寸设置

## 调试建议

1. **参数递进测试**：从最保守设置开始，逐步调整单个参数
2. **对比分析**：保存不同参数组合的结果进行对比
3. **重点区域检查**：特别关注面部、手部等关键区域
4. **批量测试**：使用相同参数处理多张图片验证稳定性

通过以上预设配置，可以有效控制IC-Light处理过程中背景和文本对前景图像的影响程度，确保在获得理想光照效果的同时最大程度保持前景的原始特征。

## 影响文本条件的参数详解

### 直接影响文本的核心参数

#### 1. **CFG Scale (`cfg`)**
- **影响程度**：最高
- **作用机制**：控制文本提示的引导强度，是Classifier-Free Guidance的核心参数
- **数值范围**：1.0-20.0
- **效果分析**：
  - **低值(1-3)**：文本影响极弱，主要保持原图特征
  - **中低值(3-5)**：轻微文本引导，适合保守调整
  - **中等值(5-8)**：平衡的文本影响，推荐范围
  - **高值(8-15)**：强烈文本引导，效果明显
  - **极高值(15-20)**：过度引导，可能产生伪影
- **推荐设置**：
  - 保持前景：2-5
  - 平衡效果：5-8
  - 强化文本：8-12

#### 2. **扩散步数 (`steps`)**
- **影响程度**：高
- **作用机制**：控制文本条件的迭代优化次数，影响文本融合的充分程度
- **数值范围**：10-50
- **效果分析**：
  - **少步数(10-15)**：快速生成，文本影响相对较弱
  - **中等步数(20-30)**：充分优化，文本条件较好融入
  - **多步数(35-50)**：过度优化，可能过拟合文本条件
- **推荐设置**：
  - 快速预览：10-15
  - 标准处理：20-25
  - 精细调整：25-35

#### 3. **高分辨率去噪强度 (`highres_denoise`)**
- **影响程度**：高
- **作用机制**：在高分辨率阶段重新应用文本条件，二次强化文本影响
- **数值范围**：0.1-1.0
- **效果分析**：
  - **低值(0.1-0.3)**：保持低分辨率结果，文本影响有限
  - **中等值(0.3-0.6)**：适度重新生成，平衡质量与保真度
  - **高值(0.7-1.0)**：大幅重新生成，文本条件影响显著
- **推荐设置**：
  - 保持原貌：0.1-0.3
  - 平衡处理：0.3-0.5
  - 强化文本：0.6-0.8

### 文本内容相关参数

#### 4. **主要提示词 (`prompt`)**
- **影响程度**：最高
- **作用**：定义期望的光照效果、风格和氛围
- **关键词分类**：
  - **光照类型**："natural lighting", "studio lighting", "rim lighting", "ambient light"
  - **光照质量**："soft", "harsh", "dramatic", "subtle", "gentle"
  - **光照方向**："from left", "from above", "backlighting", "side lighting"
  - **氛围描述**："warm", "cool", "cinematic", "professional", "artistic"
- **文本影响强度**：
  - **弱影响词汇**："natural", "subtle", "gentle", "soft"
  - **中等影响词汇**："professional", "balanced", "enhanced"
  - **强影响词汇**："dramatic", "cinematic", "artistic", "stylized"

#### 5. **附加提示词 (`a_prompt`)**
- **影响程度**：中等
- **作用**：补充质量描述和风格细节
- **常用模板**：
  - **质量提升**："high quality, detailed, sharp, professional"
  - **摄影风格**："professional photography, studio shot, portrait"
  - **技术规格**："8k, ultra detailed, masterpiece"
- **使用策略**：
  - 保守处理：只添加质量词汇
  - 风格增强：添加具体风格描述
  - 技术优化：添加分辨率和质量要求

#### 6. **负面提示词 (`n_prompt`)**
- **影响程度**：中等
- **作用**：避免不想要的效果，间接控制文本影响
- **关键避免词汇**：
  - **变形相关**："face distortion", "morphing", "deformed", "disfigured"
  - **光照问题**："overexposed", "underexposed", "harsh shadows", "blown out"
  - **风格问题**："cartoon", "anime", "painting", "artistic effects"
  - **质量问题**："blurry", "low quality", "pixelated", "artifacts"
- **模板示例**：
```
# 保守型负面提示
face distortion, morphing, overexposed, harsh lighting, unnatural colors

# 全面型负面提示
face distortion, facial morphing, style transfer, dramatic lighting, 
overexposed, underexposed, color cast, unnatural colors, 
artistic effects, painting style, cartoon, anime, 
strong shadows, harsh lighting, extreme contrast, 
blurry, low quality, artifacts, deformed
```

### 间接影响文本的参数

#### 7. **高分辨率缩放 (`highres_scale`)**
- **影响程度**：中等
- **作用机制**：影响文本条件在高分辨率阶段的应用效果
- **影响分析**：
  - 较大缩放会在高分辨率阶段增强文本影响
  - 配合`highres_denoise`共同作用
- **推荐设置**：
  - 保守处理：1.2-1.5
  - 标准处理：1.5-2.0
  - 强化处理：2.0-2.5

#### 8. **图像尺寸设置**
- **影响程度**：较低
- **作用机制**：不同尺寸下文本条件的表现可能有差异
- **注意事项**：
  - 保持合理的宽高比
  - 避免过小或过大的尺寸
  - 考虑模型训练时的标准尺寸

### 文本影响强度控制策略

#### 最小文本影响配置：
```yaml
cfg: 2-4
steps: 10-15
highres_denoise: 0.1-0.25
highres_scale: 1.0-1.3
prompt: "natural lighting, preserve original"
a_prompt: "high quality, maintain features"
n_prompt: "face distortion, dramatic effects, style change"
```

#### 平衡文本影响配置：
```yaml
cfg: 5-7
steps: 20-25
highres_denoise: 0.3-0.5
highres_scale: 1.5-2.0
prompt: "soft professional lighting"
a_prompt: "high quality, detailed, balanced"
n_prompt: "harsh lighting, extreme effects, distortion"
```

#### 强化文本影响配置：
```yaml
cfg: 8-12
steps: 25-35
highres_denoise: 0.6-0.8
highres_scale: 2.0-2.5
prompt: "cinematic dramatic lighting, artistic enhancement"
a_prompt: "professional photography, stylized, enhanced"
n_prompt: "low quality, artifacts, overexposed"
```

### 文本条件的技术实现机制

#### 1. **CLIP文本编码**
- 文本通过CLIP模型编码为768维向量
- 支持最大77个token的文本长度
- 编码过程保留语义信息和上下文关系

#### 2. **Cross-Attention机制**
- 文本向量通过交叉注意力机制影响UNet的每一层
- 在不同分辨率层级都有文本条件注入
- 注意力权重决定文本对图像不同区域的影响强度

#### 3. **Classifier-Free Guidance**
- CFG通过对比有条件和无条件生成来增强文本引导
- 公式：`output = unconditional + cfg_scale * (conditional - unconditional)`
- CFG Scale直接控制文本条件的影响强度

#### 4. **多尺度文本应用**
- 低分辨率阶段：建立基本的光照和风格
- 高分辨率阶段：根据`highres_denoise`重新应用文本条件
- 两阶段协同确保文本效果的一致性和细节质量

### 文本优化实用技巧

#### 1. **文本权重控制**
- 使用括号增强权重：`(dramatic lighting:1.2)`
- 使用方括号减弱权重：`[subtle:0.8]`
- 权重范围通常在0.5-1.5之间

#### 2. **文本组合策略**
- 主要效果 + 质量描述：`"rim lighting, high quality"`
- 具体描述 + 保护性词汇：`"soft studio lighting, preserve facial features"`
- 分层描述：光照类型 → 光照质量 → 技术要求

#### 3. **A/B测试方法**
- 固定其他参数，只改变文本内容
- 记录不同文本描述的效果差异
- 建立个人的文本效果数据库

#### 4. **文本调试流程**
1. 从简单描述开始：`"good lighting"`
2. 逐步添加具体描述：`"soft rim lighting"`
3. 加入质量要求：`"soft rim lighting, professional"`
4. 优化负面提示：避免不想要的效果
5. 微调权重和CFG Scale

### 文本优化实用技巧

#### 1. **文本权重控制**
- 使用括号增强权重：`(dramatic lighting:1.2)`
- 使用方括号减弱权重：`[subtle:0.8]`
- 权重范围通常在0.5-1.5之间

#### 2. **文本组合策略**
- 主要效果 + 质量描述：`"rim lighting, high quality"`
- 具体描述 + 保护性词汇：`"soft studio lighting, preserve facial features"`
- 分层描述：光照类型 → 光照质量 → 技术要求

#### 3. **A/B测试方法**
- 固定其他参数，只改变文本内容
- 记录不同文本描述的效果差异
- 建立个人的文本效果数据库

#### 4. **文本调试流程**
1. 从简单描述开始：`"good lighting"`
2. 逐步添加具体描述：`"soft rim lighting"`
3. 加入质量要求：`"soft rim lighting, professional"`
4. 优化负面提示：避免不想要的效果
5. 微调权重和CFG Scale

---

# 扩散生成阶段：降低背景对前景影响的技术策略

## 核心原理

在IC-Light的扩散生成过程中，背景图像通过VAE编码后与前景图像拼接，作为UNet的强条件输入。要降低背景对前景的影响，需要在扩散过程的不同阶段采用针对性策略。

## 关键参数控制策略

### 1. 扩散步数优化

**步数选择策略**：
- **快速处理（15-25步）**：适合预览和快速测试，背景影响相对较小
- **标准质量（25-35步）**：平衡处理速度和效果质量，推荐日常使用
- **高质量输出（35-50步）**：获得最佳光照融合效果，但处理时间较长

**步数影响分析**：
- 更多步数能提供更精细的光照细节和更自然的前景背景融合
- 较少步数可以减少背景对前景的累积影响，但可能牺牲光照效果
- 建议根据具体需求在质量和速度之间找到平衡点

### 2. CFG Scale固定值调节

**CFG Scale全程固定策略**：
- **保守设置（2-4）**：最小化背景和文本对前景的影响，适合前景保持优先
- **平衡设置（5-8）**：在光照效果和前景保持之间取得平衡，推荐大多数场景
- **强化设置（8-12）**：强化光照效果和文本引导，适合需要明显光照变化的场景
- **极值设置（12+）**：极强的引导效果，可能导致前景过度变化，谨慎使用

**CFG与背景影响的关系**：
```
低CFG (1-4)：背景影响最小，但光照效果弱
中CFG (5-8)：平衡背景融合与前景保持
高CFG (9-15)：背景影响强，可能过度改变前景
```

### 3. 高分辨率去噪强度控制

**Highres Denoise固定值策略**：
- **保守去噪（0.2-0.4）**：最小化第二阶段的变化，主要用于分辨率提升
- **标准去噪（0.5-0.7）**：平衡细节优化和稳定性，推荐设置
- **强化去噪（0.7-0.9）**：允许更多的细节重建，可能带来更好的光照效果
- **极值去噪（0.9+）**：接近完全重新生成，风险较高

**去噪强度影响**：
- 较低值主要进行分辨率提升，保持第一阶段的结果
- 较高值允许更多的细节重建和光照优化
- 需要与CFG Scale协调使用以获得最佳效果

## 高级技术策略

### 1. 输入预处理优化

**前景图像预处理**：
- 确保前景图像具有清晰的边缘和良好的对比度
- 使用高质量的背景移除工具获得干净的前景
- 保持前景图像的原始分辨率和细节

**背景图像选择**：
- 选择与前景主体尺度匹配的背景
- 避免背景中有过于复杂或抢眼的元素
- 确保背景的光照方向与期望的最终效果一致

### 2. 提示词策略优化

**正向提示词构建**：
- 重点描述期望的光照效果和氛围
- 避免过于详细的前景描述，以免干扰原有特征
- 使用光照相关的关键词："soft lighting", "natural light", "ambient lighting"

**负向提示词应用**：
- 添加防止前景变形的关键词："deformed", "distorted", "blurry"
- 包含防止背景过度影响的词汇："background bleeding", "color contamination"
- 使用通用的质量控制词汇提升整体效果

### 3. 多阶段处理策略

**渐进式参数调整**：
- 首先使用保守参数进行测试（低CFG，中等步数）
- 根据初步结果调整参数强度
- 在高分辨率阶段进行细节优化

**批量处理优化**：
- 对同类型的前景使用相似的参数设置
- 建立参数模板库，提高处理效率
- 记录成功的参数组合，形成最佳实践库

## 实用参数组合

### 最小背景影响设置（前景保持优先）
```
CFG Scale: 3-5
Steps: 15-25
Highres Denoise: 0.3-0.5
Highres Scale: 1.5-2.0
Seed: 固定值（便于对比调试）
```
**适用场景**：需要最大程度保持前景原貌，只进行轻微的光照调整

### 平衡融合设置（推荐日常使用）
```
CFG Scale: 6-8
Steps: 25-35
Highres Denoise: 0.5-0.7
Highres Scale: 2.0
Seed: 随机或固定
```
**适用场景**：在光照效果和前景保持之间取得良好平衡，适合大多数应用

### 强光照融合设置（效果优先）
```
CFG Scale: 8-12
Steps: 35-50
Highres Denoise: 0.7-0.9
Highres Scale: 2.0-2.5
Seed: 多次尝试选择最佳
```
**适用场景**：需要显著的光照变化和强烈的氛围效果，可接受前景的适度变化

## 监控与调试

### 关键指标
1. **前景保真度**：对比原始前景的结构相似性
2. **光照一致性**：检查光照方向和强度的合理性
3. **边缘质量**：观察前景边缘的自然度
4. **色彩协调**：评估整体色彩的和谐性

### 调试流程
1. **基线测试**：使用标准参数生成基准结果
2. **单参数变化**：逐个调整参数观察影响
3. **组合优化**：找到最佳参数组合
4. **批量验证**：在多个样本上验证效果稳定性

### 常见问题解决

**前景过度变化**：
- 降低CFG Scale到3-5
- 减少扩散步数到15-25
- 降低高分辨率去噪强度到0.3-0.5
- 优化负向提示词，添加防变形关键词

**光照效果不足**：
- 适当提高CFG Scale到7-10
- 增加扩散步数到30-40
- 提高高分辨率去噪强度到0.6-0.8
- 优化正向提示词，强化光照描述

**边缘不自然**：
- 调整高分辨率去噪强度到0.4-0.6
- 确保前景图像边缘清晰
- 选择合适的高分辨率缩放比例
- 使用高质量的前景抠图

**整体色彩不协调**：
- 调整CFG Scale到中等值（5-8）
- 在提示词中添加色彩协调描述
- 选择色调匹配的背景图像
- 适当调整高分辨率去噪强度

通过深入理解这些文本相关参数的作用机制和调整策略，可以精确控制IC-Light中文本条件对图像生成的影响，实现理想的光照效果同时保持前景特征的完整性。