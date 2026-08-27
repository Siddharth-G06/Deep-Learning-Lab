# Experiment 5: Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

**Course:** CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai  
**Academic Year:** 2026–27  
**Semester:** V  

---

## Objective

The objective of this experiment is to systematically study the effect of weight initialization, regularization, optimization algorithms, CNN hyperparameters, transfer learning, fine-tuning and cross-validation on image classification performance.

The experiment uses a single CNN architecture, MobileNetV2, and the Oxford-IIIT Pet dataset. Students will analyze the effect of individual design choices using training/validation curves and finally select a suitable configuration using 5-fold cross-validation.

---

## Learning Outcomes

After completing this experiment, students will be able to:
- explain different weight initialization strategies;
- identify and analyze overfitting using training and validation curves;
- compare SGD, Momentum, RMSProp and Adam;
- explain CNN hyperparameters such as kernel size, stride, padding and pooling;
- understand Batch Normalization and Dropout;
- perform transfer learning and fine-tuning using MobileNetV2;
- tune important training hyperparameters;
- apply 5-fold cross-validation for model selection; and
- justify the final model using accuracy, variability and computational cost.

---

## Dataset Information

**Dataset Name:** Oxford-IIIT Pet Dataset

The Oxford-IIIT Pet Dataset contains images of cats and dogs belonging to 37 breeds. The images are RGB and have different spatial dimensions. For this experiment, all images are resized to 224 × 224 × 3.

The dataset is suitable for demonstrating transfer learning using ImageNet-pretrained CNNs.

**Experimental considerations:**
- Use RGB images resized to 224 × 224.
- Normalize the input images according to the pretrained model requirements.
- Maintain separate training, validation and test data.
- The test set must remain untouched during hyperparameter selection.
- Use MobileNetV2 because it is computationally lighter than many large CNN architectures and is more suitable for CPU-based laboratory work.

**Dataset Split:** The Oxford-IIIT Pet training split of 3680 images was divided into 2944 training images and 736 validation images (80:20 split), while the official test split of 3669 images was kept completely independent and used only for final evaluation.

---

## Repository Contents

```
Lab 5 - Comprehensive Study of CNN Training/
├── README.md                            # This document
├── dl-lab-5.ipynb                       # Jupyter Notebook with the source code
└── images/                              # Images folder for the lab
```

---

## Dependencies

The following Python libraries are required to run the code:
- `tensorflow`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `jupyter`

To install these dependencies, run the following command in your terminal:
```bash
pip install -r requirements.txt
```
*(Note: Create a requirements.txt if not already present or install packages manually)*

---

## Execution Instructions

1. Open your terminal and navigate to the experiment folder:
   ```bash
   cd "Lab 5 - Comprehensive Study of CNN Training"
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook dl-lab-5.ipynb
   ```
3. Execute the cells sequentially from top to bottom.
