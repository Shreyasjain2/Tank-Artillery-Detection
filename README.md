# Parameter-Efficient Fine-Tuning of Florence-2 for Military Object Detection

## Overview
This repository presents a **parameter-efficient adaptation of Florence-2**, a large vision–language foundation model, for **military object detection**, focusing on **tanks, artillery, and armored vehicles**.  
By leveraging **Low-Rank Adaptation (LoRA)**, the model is specialized for the defense domain using limited labeled data while preserving its strong zero-shot and general-purpose capabilities.

The system processes **multimodal inputs (image + text prompt)** and generates **precise object localizations in textual form**, enabling flexible post-processing and deployment in constrained environments.

---

## Key Features
- Prompt-based **vision–language object detection**
- **LoRA-based fine-tuning** with <1% trainable parameters
- Textual generation of bounding box coordinates
- Designed for **data-scarce and compute-constrained** military settings
- Compatible with RGB, IR, and SAR imagery

---

## Florence-2 Architecture

Florence-2 is a unified vision–language foundation model that treats all vision tasks as **text generation problems**.

### High-Level Architecture
![Florence-2 Architecture](Screenshots/florence2_architecture.png)

### Core Components
- **Vision Encoder**
  - Transformer-based (e.g., ViT-H/16)
  - Extracts hierarchical spatial features from images
- **Text Decoder**
  - Autoregressive transformer
  - Generates object labels and bounding box coordinates as text
- **Multimodal Fusion**
  - Cross-attention aligns visual embeddings with textual prompts
- **Prompt-Based Design**
  - Tasks are specified using natural language prompts  
    Example:
    ```
    Detect military objects in this image:
    ```

---

## Parameter-Efficient Fine-Tuning with LoRA

Instead of full fine-tuning, we inject **Low-Rank Adaptation (LoRA)** layers into selected transformer modules.

### LoRA Injection Strategy
![LoRA Architecture](Screenshots/lora_architecture.png)

### Advantages
- 10–100× reduction in trainable parameters
- Avoids catastrophic forgetting
- Faster convergence on small datasets
- Lower GPU memory requirements

### Configuration
| Parameter | Value |
|--------|------|
| Rank (r) | 8 |
| Alpha | 8 |
| Dropout | 0.05 |
| Target Modules | Attention projections, FFN layers, vision Conv2D, lm_head |

---

## Dataset Design

The dataset is stored in **JSONL format**, pairing images with prompt–response text.

### Dataset Structure
![Dataset Structure](Screenshots/dataset_structure.png)

### JSONL Sample
```json
{
  "image": "tank_001.jpg",
  "prefix": "Detect military objects in this image:",
  "suffix": "Tank (x1, y1, x2, y2); Artillery (x1, y1, x2, y2)"
}
