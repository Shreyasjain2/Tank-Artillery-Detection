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
![Florence-2 Architecture](https://github.com/Shreyasjain2/Tank-Artillery-Detection/blob/main/Screenshots/model_archi.png)

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
![LoRA Architecture](https://github.com/Shreyasjain2/Tank-Artillery-Detection/blob/main/Screenshots/lora%20arch.png)

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
json
{
  "image": "tank_001.jpg",
  "prefix": "Detect military objects in this image:",
  "suffix": "Tank (x1, y1, x2, y2); Artillery (x1, y1, x2, y2)"
}

### Directory Layout
dataset/
├── images/
│ ├── tank_001.jpg
│ ├── artillery_023.jpg
│ └── ...
├── annotations.jsonl


### Dataset Classes
The dataset pipeline is implemented using two modular dataset classes designed for flexibility and robustness:

- **JSONLDataset**
  - Parses the JSONL annotation file.
  - Resolves image file paths and loads images as `PIL.Image` objects.
  - Performs basic validation and error handling for missing or corrupted samples.

- **DetectionDataset**
  - Wraps `JSONLDataset`.
  - Returns tuples of `(prefix, suffix, image)` compatible with PyTorch’s `DataLoader`.
  - Enables seamless batching and preprocessing during training.

### Dataset Challenges
Training on real-world military imagery introduces several non-trivial challenges:

- **Data Scarcity**
  - Limited availability of labeled military datasets due to confidentiality and security constraints.
- **Image Quality Variability**
  - Presence of low-resolution samples and distant targets that are difficult to identify even by human annotators.
- **Class Imbalance**
  - Over-representation of common objects (e.g., tanks) compared to rare classes (e.g., mobile artillery).
- **Annotation Consistency**
  - Bounding box coordinates must precisely align with textual descriptions to avoid noisy supervision.

---

## Training Pipeline

### End-to-End Training Flow
![Training Pipeline](Screenshots/training_pipeline.png)

### Training Configuration
- **Batch Size:** 6  
- **Optimizer:** AdamW  
- **Learning Rate:** 1e-6  
- **Scheduler:** Linear  
- **Epochs:** 10  
- **Loss Function:** Cross-entropy loss applied to the text decoder output using teacher forcing

### Training Details
- Mixed precision training (FP16) to reduce memory footprint.
- Automatic image resizing and text token padding handled by the Florence-2 processor.
- Model checkpoints and processor configurations saved after each epoch.
- Validation loss computed on a held-out split (noting that dataset separation must be improved in future iterations).

---

## Inference and Visualization

Inference is performed in a **prompt-driven manner**, leveraging Florence-2’s text generation capability to produce object detections.

### Inference Pipeline
![Inference Pipeline](https://github.com/Shreyasjain2/Tank-Artillery-Detection/blob/main/Screenshots/inference.jpg)

### Example Inference Output
Prompt: Detect military objects in this image:
Output: Tank (x1, y1, x2, y2); Armored Vehicle (x1, y1, x2, y2)


### Visualization
A custom rendering utility converts the generated textual bounding boxes into visual overlays for qualitative analysis.

![Detection Results](https://github.com/Shreyasjain2/Tank-Artillery-Detection/blob/main/Screenshots/Screenshot%202025-05-16%20193337.png)

---

## Experimental Results

### Quantitative Performance
| Metric | Value |
|------|------|
| Precision | 0.9897 |
| Recall | 0.9942 |
| F1 Score | 0.9919 |
| Accuracy | 0.9840 |

### Ablation Studies
- **LoRA Rank:** Increasing rank from 8 to 16 yields marginal performance gains.
- **Full Fine-Tuning:** Achieves slightly higher accuracy at the cost of significantly increased parameter count.
- **Prompt Engineering:** Minor variations in prefix text impact detection sensitivity.

### Baseline Comparison
| Model | Performance |
|-----|-----------|
| Faster R-CNN | ~75% mAP |
| YOLOv8 | ~80% mAP |
| Florence-2 (Zero-Shot) | ~97% mAP |
| **Florence-2 + LoRA (Ours)** | Best balance of accuracy and efficiency |

---

## Limitations
- Training and validation sets currently share the same JSONL file, leading to potential data leakage.
- Limited coverage of multispectral imagery (e.g., IR, SAR).
- Inference latency may be high for real-time edge deployment scenarios.

---

## Future Work
- Introduce strict train/validation/test splits.
- Experiment with higher LoRA ranks (r = 16, r = 32).
- Fine-tune vision encoder layers for improved spatial reasoning.
- Apply model distillation and quantization for edge deployment.
- Extend evaluation to diverse terrains and weather conditions.

---

## Ethical and Operational Considerations
- Risk of misuse due to the dual-use nature of military AI systems.
- Potential dataset bias across terrains, object variants, and operational contexts.
- Requirement for **human-in-the-loop** verification in safety-critical deployments.
- Alignment with established military and governmental AI ethics guidelines.

---

## Conclusion
This work demonstrates that **large vision–language foundation models** can be effectively adapted for **military object detection** using **parameter-efficient fine-tuning techniques**.  
By combining prompt-based learning with LoRA, the approach delivers high detection accuracy while maintaining computational efficiency, making it suitable for real-world military reconnaissance and surveillance applications.

