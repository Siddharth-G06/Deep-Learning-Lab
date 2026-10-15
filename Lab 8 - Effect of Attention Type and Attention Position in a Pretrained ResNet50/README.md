# Lab 8 — Effect of Attention Type and Attention Position in a Pretrained ResNet50

> **Shiv Nadar University Chennai | CS3807 — Deep Learning Laboratory | Academic Year 2026–27**

---

## Experiment Overview

This experiment investigates how different attention mechanisms modify a pretrained convolutional neural network and how the position at which attention is inserted influences image classification performance. A single pretrained architecture — **ResNet50 pretrained on ImageNet** — is used throughout. Five types of flower images from the TensorFlow Flowers dataset are classified, and systematic ablation studies are performed across attention types, insertion positions, ordering, kernel sizes, and multi-stage configurations.

---

## Objective

The objective is to:

- Understand the stage-wise feature hierarchy of a pretrained ResNet50.
- Implement channel attention, spatial attention, sequential channel–spatial attention, and lightweight self-attention.
- Insert each attention module at different stages (P1–P4) of the ResNet50 and compare performance.
- Study the effect of attention ordering (channel→spatial vs. spatial→channel).
- Evaluate the spatial attention kernel size effect (3×3, 5×5, 7×7).
- Analyse multi-stage attention placement (P3 only, P4 only, P3+P4).
- Generate and interpret Grad-CAM visualizations and internal attention maps.
- Produce a comprehensive accuracy–complexity trade-off analysis.

---

## Learning Outcomes

After completing this experiment, students will be able to:

1. Explain the stage-wise architecture of a pretrained ResNet50.
2. Distinguish early, intermediate, and late convolutional features.
3. Explain the motivation behind attention mechanisms.
4. Implement channel attention (SE-style).
5. Implement spatial attention.
6. Implement sequential channel–spatial attention (CBAM-style).
7. Implement a lightweight self-attention mechanism.
8. Insert an attention module at different stages of a pretrained network.
9. Analyse the influence of attention position.
10. Compare different attention mechanisms under a controlled experimental protocol.
11. Analyse classification accuracy and class-wise performance.
12. Visualise internal attention maps.
13. Generate Grad-CAM explanations.
14. Distinguish an internal attention map from a post-hoc explanation map.
15. Analyse accuracy–complexity trade-offs.
16. Perform controlled ablation studies.

---

## Dataset

### TensorFlow Flowers (`tf_flowers`)

| Property | Value |
|---|---|
| **Classes** | 5 (daisy, dandelion, roses, sunflowers, tulips) |
| **Total Images** | 3,670 |
| **Input Size** | 224 × 224 × 3 (resized) |
| **Preprocessing** | ResNet50 `preprocess_input` (channel-wise mean subtraction) |
| **Split** | 70% train / 15% validation / 15% test |
| **Source** | `tensorflow_datasets` (`tfds.load("tf_flowers")`) |

The dataset is streamed from disk without a large in-memory shuffle buffer to minimize RAM usage.

---

## ResNet50 Feature Hierarchy

| Stage | Location | Spatial Resolution | Channels |
|---|---|---|---|
| Stem + MaxPool | — | 56 × 56 | 64 |
| conv2_x | P1 | 56 × 56 | 256 |
| conv3_x | P2 | 28 × 28 | 512 |
| conv4_x | P3 | 14 × 14 | 1024 |
| conv5_x | P4 | 7 × 7 | 2048 |

Attention position **P3** (after `conv4_block6_out`) is used as the default insertion point for type-ablation experiments.

---

## Attention Mechanisms

### 1. Channel Attention (SE-style)
Determines the relative importance of each feature channel using global average pooling followed by two FC layers with a ReLU–Sigmoid pathway. Reduction ratio `r = 16`.

```
F → GlobalAvgPool → FC(C/r, relu) → FC(C, sigmoid) → scale F
```

### 2. Spatial Attention
Determines the relative importance of each spatial location by pooling across channels and applying a convolution.

```
F → [AvgPool_c || MaxPool_c] → Conv(k×k) → sigmoid → scale F
```

Kernel size `k ∈ {3, 5, 7}` (default: 7).

### 3. Sequential Channel–Spatial Attention (CBAM-style)
Applies channel attention first, then spatial attention on the channel-refined features.

```
F → Channel Attention → Spatial Attention → F_cs
```

### 4. Lightweight Self-Attention
Computes Q, K, V projections from flattened spatial features and applies scaled dot-product attention with a residual connection:

```
F → flatten → QKV projections → Attention(QK^T/sqrt(d)) · V → reshape → F + γ·Y
```

Self-attention is computationally intensive (O(N²) with N = H×W), so it is preferred at P3 or P4 where spatial resolution is small.

---

## Experimental Studies

### Experiment 1 — Attention Type Ablation (Fixed position P3)

| Model | Channel | Spatial | Self-Attention |
|---|---|---|---|
| A0: Baseline ResNet50 | – | – | – |
| A1: Channel Attention | ✓ | – | – |
| A2: Spatial Attention | – | ✓ | – |
| A3: Channel + Spatial | ✓ | ✓ | – |
| A4: Self-Attention | – | – | ✓ |
| A5: Combined (All) | ✓ | ✓ | ✓ |

**Research question:** How does attention *type* affect model performance?

### Experiment 2 — Attention Position Ablation (Fixed type: Channel+Spatial)

| Config | Position | Resolution | Channels |
|---|---|---|---|
| B1 | After conv2_x (P1) | 56 × 56 | 256 |
| B2 | After conv3_x (P2) | 28 × 28 | 512 |
| B3 | After conv4_x (P3) | 14 × 14 | 1024 |
| B4 | After conv5_x (P4) | 7 × 7 | 2048 |

**Research question:** How does the *location* of attention influence performance?

### Experiment 3 — Multi-Stage Attention

Attention at P3 only vs. P4 only vs. P3 + P4 (extended: P2 + P3 + P4).

**Research question:** Is attention at more locations necessarily more effective?

### Experiment 4 — Attention Ordering

At P3: Channel → Spatial (`F → Mc → Ms`) vs. Spatial → Channel (`F → Ms → Mc`).

**Research question:** Does the order in which channel and spatial attention are applied matter?

### Experiment 5 — Spatial Attention Kernel Size

Spatial kernel `k ∈ {3×3, 5×5, 7×7}` at a fixed position.

**Research question:** Does a larger spatial receptive field improve attention quality?

### Additional Study — Comparison of Established Attention Mechanisms

Comparison of custom implementations with well-known modules: SE, ECA, CBAM, Coordinate Attention, and lightweight Self-Attention, all inserted at P3.

---

## Experimental Procedure

The full implementation is documented in `Lab_8_final.ipynb` and follows these tasks:

1. **Environment Setup:** Configure GPU memory growth, install dependencies, set random seeds.
2. **Data Pipeline:** Load `tf_flowers` using TFDS, apply 70/15/15 split, preprocess with ResNet50 `preprocess_input`, batch (size 8), prefetch.
3. **Hyperparameter Search:** Grid search over learning rate ∈ {1e-3, 1e-4, 1e-5}, optimizer ∈ {Adam, SGD}, dropout ∈ {0.3, 0.5} using 3-epoch trials.
4. **Baseline Model:** Train ResNet50 (frozen backbone) + GAP + Dropout + Dense(5) with the best found hyperparameters. Record accuracy, loss curves, confusion matrix.
5. **SE / Channel Attention at P3:** Insert SE block after `conv4_block6_out`, train and evaluate.
6. **Spatial Attention at P3:** Insert spatial attention block, train and evaluate.
7. **CBAM at P2, P3, P4:** Insert full CBAM (channel + spatial) at each position, compare.
8. **Self-Attention at P3:** Insert lightweight self-attention, train and evaluate.
9. **Multi-Stage Attention:** P3 only, P4 only, P3+P4, evaluate each.
10. **Attention Ordering:** Channel→Spatial vs. Spatial→Channel at P3.
11. **Spatial Kernel Study:** Compare k=3, k=5, k=7 for spatial attention.
12. **Grad-CAM Visualization:** Generate Grad-CAM maps for baseline vs. best attention model.
13. **Consolidated Results:** Plot bar charts for accuracy, macro F1, parameters, inference time.

---

## Required Plots

| Plot Filename | Description |
|---|---|
| `baseline_accuracy.eps` | Baseline ResNet50 training & validation accuracy |
| `baseline_loss.eps` | Baseline ResNet50 training & validation loss |
| `baseline_confusion_matrix.eps` | Baseline confusion matrix |
| `se_p3_training_curves.eps` | SE attention at P3 training curves |
| `baseline_vs_se_p3.eps` | Baseline vs SE (P3) bar comparison |
| `spatial_p3_training_curves.eps` | Spatial attention at P3 training curves |
| `baseline_se_spatial_comparison.eps` | Baseline vs SE vs Spatial comparison |
| `cbam_p3_accuracy.eps` | CBAM at P3 accuracy curves |
| `cbam_p3_loss.eps` | CBAM at P3 loss curves |
| `cbam_p3_confusion_matrix.eps` | CBAM at P3 confusion matrix |
| `cbam_position_comparison.eps` | CBAM at P2 vs P3 vs P4 comparison |
| `spatial_attention_p2_p3_p4.eps` | Spatial attention position comparison |
| `gradcam_baseline_vs_cbam_p3.eps` | Grad-CAM: Baseline vs CBAM at P3 |
| `all_models_confusion_matrices.eps` | All model confusion matrices |
| `training_accuracy_comparison.eps` | Training accuracy curves across all models |
| `attention_type_comparison.eps` | Attention type ablation bar chart |
| `all_model_performance_comparison.eps` | Full model performance comparison |
| `self_attention_p3_training_curves.eps` | Self-attention training curves |
| `self_attention_p3_confusion_matrix.eps` | Self-attention confusion matrix |
| `inference_time_comparison.eps` | Inference time per model |
| `parameter_comparison.eps` | Trainable parameter count comparison |
| `dataset_samples.eps` | Sample images from the dataset |
| `accuracy_vs_inference_time.eps` | Accuracy–complexity scatter plot |
| `attention_order_comparison.eps` | Channel→Spatial vs Spatial→Channel |
| `multi_stage_attention_comparison.eps` | Multi-stage attention results |
| `combined_training_validation_curves.eps` | All models combined training curves |
| `accuracy_vs_model_complexity.eps` | Accuracy vs parameter count |
| `four_model_performance.eps` | Key four-model comparison |
| `key_metrics_comparison.eps` | Key metrics bar chart |

---

## Repository Contents

```
Lab 8 - Effect of Attention Type and Attention Position in a Pretrained ResNet50/
+-- README.md                  # This document
+-- Lab_8_final.ipynb          # Jupyter Notebook with the full implementation
+-- Experiment_8.tex           # Lab manual (LaTeX source — placeholder)
+-- Academics_SNUC.pdf         # Lab manual (PDF)
```

---

## Required Result Table

| Model | Position | Accuracy | Precision | Recall | F1 | Parameters | Inference Time |
|---|---|---|---|---|---|---|---|
| Baseline ResNet50 | – | — | — | — | — | — | — |
| Channel Attention | P3 | — | — | — | — | — | — |
| Spatial Attention | P3 | — | — | — | — | — | — |
| Channel+Spatial | P2 | — | — | — | — | — | — |
| Channel+Spatial | P3 | — | — | — | — | — | — |
| Channel+Spatial | P4 | — | — | — | — | — | — |
| Self-Attention | P3 | — | — | — | — | — | — |
| Additional Attention-1 | P3 | — | — | — | — | — | — |
| Additional Attention-2 | P3 | — | — | — | — | — | — |

*(Values are to be filled in from the student's own execution of `Lab_8_final.ipynb`.)*

---

## Key Observations

- **Attention Position Matters:** Middle-stage attention (P3) tends to balance spatial resolution and semantic depth, making it generally most effective. Early-stage attention (P1, P2) has high spatial cost but operates on low-level features; late-stage attention (P4) operates on compact but highly semantic representations.
- **Channel vs. Spatial Attention:** Channel attention (SE-style) identifies which feature maps are informative, while spatial attention highlights where in the feature map useful information lies. Their combination (CBAM) typically outperforms either alone.
- **Self-Attention Complexity:** Self-attention captures long-range spatial dependencies but grows quadratically with spatial resolution (O(N²) with N = H×W), making it most practical at P3 or P4.
- **Attention Ordering:** Channel→Spatial ordering (select *what* then *where*) generally performs slightly better than Spatial→Channel because selecting informative channels first reduces noise before spatial selection.
- **Spatial Kernel Size:** Larger kernels (7×7) provide a wider receptive field for attention map generation, which can capture broader contextual information, though at a small cost in parameters.
- **Multi-Stage Attention:** Stacking attention at multiple ResNet50 stages (P3+P4) does not always outperform single-stage insertion; the computational overhead may not be justified by marginal accuracy gains.
- **Grad-CAM Evidence:** Grad-CAM visualizations show that attention models focus more tightly on flower petals and discriminative regions compared to the baseline, confirming that attention improves feature localization.

---

## Dependencies

The following Python libraries are required to run the notebook:

- `tensorflow` (≥ 2.x) with GPU support recommended
- `tensorflow-datasets`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `psutil`
- `jupyter`

Install all dependencies with:

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib seaborn scikit-learn psutil jupyter
```

> **Note:** This notebook was developed and tested on **Kaggle** (GPU accelerator + Internet enabled). For local execution, ensure a CUDA-compatible GPU is available. The data pipeline uses `tf_flowers` from TFDS (~220 MB, downloaded on first run).

---

## Execution Instructions

1. Navigate to the experiment folder:
   ```bash
   cd "Lab 8 - Effect of Attention Type and Attention Position in a Pretrained ResNet50"
   ```
2. Install the required libraries:
   ```bash
   pip install tensorflow tensorflow-datasets numpy pandas matplotlib seaborn scikit-learn psutil jupyter
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Lab_8_final.ipynb
   ```
4. Execute the cells sequentially from top to bottom.
   - Enable **GPU accelerator** and **Internet** in session options (Kaggle) or configure local GPU beforehand.
   - Generated plots are saved to `lab8_plots/` in the working directory.
   - Experiment 5 (kernel size study) saves per-kernel checkpoints and can resume from `kernel_size_results.csv` if interrupted.

> **Note:** The full grid search (`RUN_GRID_SEARCH = True`) can be slow. Set `USE_SAVED_BEST_PARAMS = True` to skip the grid search and use the pre-determined best hyperparameters (LR=1e-3, optimizer=SGD, dropout=0.3).

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
2. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *CVPR*.
3. Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
4. Wang, Q., et al. (2020). ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. *CVPR*.
5. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
7. TensorFlow Datasets — tf_flowers: https://www.tensorflow.org/datasets/catalog/tf_flowers
8. Keras Applications — ResNet50: https://keras.io/api/applications/resnet/
