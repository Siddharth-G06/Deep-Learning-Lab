# Experiment 1: Single Layer Perceptron

**Course:** CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

---

## Objective

The primary objective of this experiment is to implement a Single Layer Perceptron from scratch to solve a binary classification problem. By building the model without relying on pre-built machine learning estimators, we aim to understand how an artificial neuron functions. This includes observing how weights are updated, understanding the role of the step activation function, and analyzing the impact of the learning rate on the model's performance.

---

## Dataset Information

**Dataset Name:** Banknote Authentication Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

This dataset contains 1,372 samples with 4 numerical features. These features were extracted from images of genuine and forged banknotes using the Wavelet Transform technique.

| Feature | Description |
|---------|-------------|
| `variance` | Variance of the wavelet-transformed image |
| `skewness` | Skewness of the wavelet-transformed image |
| `curtosis` | Curtosis of the wavelet-transformed image |
| `entropy` | Entropy of the image |
| `class` | Target label (0 = Authentic, 1 = Forged) |

- **Class Distribution:** 762 authentic samples and 610 forged samples.
- **Missing Values:** None.
- **File Location:** `banknote+authentication/data_banknote_authentication.txt` (Column names are assigned programmatically during data loading).

---

## Experimental Procedure

The implementation is documented in the Jupyter Notebook and is divided into the following key tasks:

1. **Dataset Exploration:** Loading the data, checking dimensions, and verifying the absence of missing values.
2. **Exploratory Data Analysis (EDA):** Visualizing the data using histograms, correlation heatmaps, scatter plots, and box plots to understand feature relationships.
3. **Data Preprocessing:** Scaling the features using Min-Max normalization and splitting the data into training (80%) and testing (20%) sets using stratified sampling.
4. **Perceptron Implementation:** Creating a `Perceptron` class from scratch that includes:
   - A step activation function.
   - Zero initialization for weights and bias.
   - The perceptron learning rule for updating weights: `w = w + lr * (y - ŷ) * x`.
   - Tracking of misclassifications, weight evolution, and bias evolution per epoch.
5. **Model Training:** Training the model for up to 100 epochs with a learning rate of 0.01.
6. **Model Evaluation:** Assessing the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. **Learning Rate Analysis:** Comparing the model's convergence behavior across different learning rates (0.001, 0.01, and 0.1).

---

## Repository Contents

```
Lab 1 - Single Layer Perceptron/
├── README.md                            ← This document
├── DL_Lab_1.ipynb                       ← Jupyter Notebook with the source code
├── requirements.txt                     ← List of required Python packages
└── banknote+authentication/
    └── data_banknote_authentication.txt ← The dataset file
```

---

## Dependencies

The following Python libraries are required to run the code:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
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
   cd "Lab 1 - Single Layer Perceptron"
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook DL_Lab_1.ipynb
   ```
4. Execute the cells sequentially from top to bottom. Generated plots will be automatically saved in a local `plots/` directory.

---

## Key Observations

- **Convergence:** The perceptron model successfully converges because the dataset classes are approximately linearly separable after feature normalization.
- **Learning Rate Impact:** A smaller learning rate (`0.001`) leads to slower, more gradual convergence, while a larger learning rate (`0.1`) can cause the model's weights to oscillate before settling.
- **Normalization:** Applying Min-Max normalization significantly improves the stability and speed of the weight updates during training.
- **Parameter Evolution:** The generated plots for weight and bias evolution clearly demonstrate how the model's parameters adjust and eventually stabilize as the training process completes.
