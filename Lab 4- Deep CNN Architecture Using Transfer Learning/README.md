# Experiment 4: Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

**Course:** CS3807 — Deep Learning Laboratory, Shiv Nadar University Chennai

---

## Objective

The objectives of this experiment are:
- Study the evolution of deep CNN architectures.
- Compare LeNet-5, AlexNet, VGG16, GoogleNet and ResNet.
- Understand transfer learning.
- Fine tune pretrained CNN models.
- Compare classification performance of different architectures.

---

## Dataset Information

**Dataset Name:** CIFAR-10 Dataset  
**Source:** [Keras Datasets](https://keras.io/api/datasets/cifar10/)

This dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

- **Training Images:** 50,000
- **Testing Images:** 10,000
- **Number of Classes:** 10
- **Image Size:** 32 × 32 × 3
- **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship and Truck.

---

## Experimental Procedure

The implementation is documented in the Jupyter Notebook and is divided into the following key tasks:

1. **Dataset Preparation:** Loading the CIFAR-10 dataset using TensorFlow/Keras, normalizing pixel values to the range [0, 1], and displaying sample images.
2. **Transfer Learning:** Loading a pretrained VGG16 model with ImageNet weights, removing the original classification layer, freezing the convolutional base, and adding a Global Average Pooling layer followed by Dense layers with ReLU and Softmax activations.
3. **Model Training:** Training the compiled model using the Adam optimizer (Learning Rate: 0.001) and Categorical Cross Entropy loss function for 10-20 epochs.
4. **Fine Tuning:** Unfreezing the last convolution block of the VGG16 model and training for additional epochs to compare classification accuracy before and after fine-tuning.
5. **Model Evaluation:** Evaluating the trained model's performance using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

## Repository Contents

```
Lab 4- Deep CNN Architecture Using Transfer Learning/
├── README.md                            # This document
├── DL_Lab_4.ipynb                       # Jupyter Notebook with the source code
├── Experiment_4_Lab_Manual.pdf          # Lab manual
├── requirements.txt                     # List of required Python packages
├── *.eps                                # Generated output plots
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

---

## Execution Instructions

1. Open your terminal and navigate to the experiment folder:
   ```bash
   cd "Lab 4- Deep CNN Architecture Using Transfer Learning"
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook DL_Lab_4.ipynb
   ```
4. Execute the cells sequentially from top to bottom. Generated plots will be automatically saved in the current directory.
