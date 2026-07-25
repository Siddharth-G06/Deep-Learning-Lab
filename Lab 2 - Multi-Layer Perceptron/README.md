# Experiment 2: Multi-Layer Perceptron (MLP) for Multi-Class Image Classification

**Course:** CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

---

## Objective

The primary objective of this experiment is to implement a Multi-Layer Perceptron (MLP) using TensorFlow/Keras on the Fashion-MNIST dataset. The experiment covers the complete deep learning workflow: image preprocessing (flattening and normalization), model construction, compilation, training, and evaluation. An additional focus is placed on automated hyperparameter optimization using Randomized Search with 5-fold cross-validation via the SciKeras wrapper, to identify the best-performing model configuration and compare it against the baseline.

---

## Dataset Information

**Dataset Name:** Fashion-MNIST  
**Source:** [Keras Built-in Dataset](https://keras.io/api/datasets/fashion_mnist/) (originally from [Zalando Research](https://github.com/zalandoresearch/fashion-mnist))

Fashion-MNIST is a dataset of Zalando's article images consisting of 70,000 grayscale images in 10 fashion categories.

| Split | Images | Shape |
|-------|--------|-------|
| Training | 60,000 | 28 × 28 |
| Testing | 10,000 | 28 × 28 |

**Class Labels:**

| Label | Class Name |
|-------|------------|
| 0 | T-shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

- **Missing Values:** None.
- **Class Distribution:** Balanced — 6,000 training samples per class.

---

## Background Theory

An MLP consists of an input layer, one or more hidden layers, and an output layer. Each layer applies a linear transformation followed by a non-linear activation function:

```
z^(l) = W^(l) * a^(l-1) + b^(l)
a^(l) = f(z^(l))
```

The output layer uses **Softmax** for multi-class classification:

```
ŷ = Softmax(z)
```

The model is trained by minimizing **Categorical Cross-Entropy** loss:

```
L = -Σ y_i * log(ŷ_i)
```

**Baseline Architecture:**

```
Input (784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```

Total parameters: **109,386** (≈ 427 KB)

---

## Experimental Procedure

The implementation in `DL_Lab_2.ipynb` is divided into the following key tasks:

1. **Dataset Exploration:** Loading Fashion-MNIST via `keras.datasets`, printing tensor shapes, displaying 10 sample images, and plotting the class distribution.
2. **Data Preprocessing:**
   - Flattening 28×28 images into 784-dimensional vectors: `reshape(-1, 784)`.
   - Normalizing pixel values to [0, 1]: `astype("float32") / 255.0`.
   - One-hot encoding labels using `to_categorical(y, num_classes=10)`.
3. **Baseline Model Construction:** Building a `Sequential` MLP with `Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)` and printing `model.summary()`.
4. **Model Training:** Compiling with Adam optimizer, Categorical Cross-Entropy loss, and Accuracy metric. Training for **20 epochs** with batch size **32** and a 20% validation split.
5. **Model Evaluation:** Assessing the baseline model using Accuracy, Precision, Recall, F1-score, a Confusion Matrix, and a full Classification Report on the test set.
6. **Hyperparameter Optimization:** Performing **Randomized Search** (10 iterations, 5-fold cross-validation) using SciKeras over the following search space:

| Hyperparameter | Candidate Values |
|----------------|-----------------|
| Hidden Layers | 1, 2, 3 |
| Hidden Neurons | 32, 64, 128, 256 |
| Learning Rate | 0.1, 0.01, 0.001 |
| Batch Size | 16, 32, 64, 128 |
| Epochs | 10, 20, 30 |
| Optimizer | Adam, SGD, RMSProp |
| Activation Function | ReLU, Tanh, Sigmoid |
| Dropout Rate | 0.0, 0.2, 0.5 |

7. **Optimized Model Training & Evaluation:** Retraining with the best hyperparameters found and comparing performance against the baseline.

---

## Repository Contents

```
LAB 2/
├── README.md                            ← This document
├── EXPERIMENT_README.md                 ← Detailed experiment README
├── DL_Lab_2.ipynb                       ← Jupyter Notebook with the source code
├── Experiment_2.tex                     ← Lab manual (LaTeX source)
├── Experiment_2_Lab_Manual.pdf          ← Lab manual (PDF)
├── Sample_Images (2).eps                ← Sample Fashion-MNIST images plot
├── Class_Distribution.eps               ← Class distribution bar chart
├── Training_Accuracy.eps                ← Training & validation accuracy curves
├── Training_Loss.eps                    ← Training & validation loss curves
├── Confusion_Matrix.eps                 ← Baseline model confusion matrix
├── Optimized_Confusion_Matrix.eps       ← Optimized model confusion matrix
├── Hyperparameter_Search.eps            ← Randomized search results visualization
└── Model_Comparison.eps                 ← Baseline vs. optimized model comparison
```

---

## Dependencies

The following Python libraries are required to run the code:

- `tensorflow` (≥ 2.x)
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scikeras`
- `jupyter`

To install these dependencies, run the following command in your terminal:
```bash
pip install tensorflow scikit-learn scikeras numpy pandas matplotlib seaborn jupyter
```

---

## Execution Instructions

1. Open your terminal and navigate to the experiment folder:
   ```bash
   cd "LAB 2"
   ```
2. Install the requirements:
   ```bash
   pip install tensorflow scikit-learn scikeras numpy pandas matplotlib seaborn jupyter
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook DL_Lab_2.ipynb
   ```
4. Execute the cells sequentially from top to bottom. Generated plots will be automatically saved as EPS files in the same directory.

> **Note:** The Randomized Search (Task 6) uses `n_jobs=-1` for parallel execution and may take several minutes depending on hardware. The search is run on a 12,000-sample subset of the training data to reduce computation time.

---

## Results

### Baseline Model Performance (Test Set)

| Metric | Score |
|--------|-------|
| Accuracy | 0.8802 |
| Precision | 0.8813 |
| Recall | 0.8802 |
| F1-Score | 0.8798 |

**Per-Class Performance (Baseline):**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| T-shirt/Top | 0.80 | 0.86 | 0.83 |
| Trouser | 0.99 | 0.96 | 0.98 |
| Pullover | 0.74 | 0.83 | 0.79 |
| Dress | 0.90 | 0.88 | 0.89 |
| Coat | 0.78 | 0.81 | 0.80 |
| Sandal | 0.97 | 0.96 | 0.97 |
| Shirt | 0.73 | 0.62 | 0.67 |
| Sneaker | 0.93 | 0.97 | 0.95 |
| Bag | 0.98 | 0.96 | 0.97 |
| Ankle Boot | 0.96 | 0.95 | 0.95 |

### Best Hyperparameters (Randomized Search)

| Hyperparameter | Best Value |
|----------------|-----------|
| Optimizer | RMSProp |
| Learning Rate | 0.001 |
| Hidden Neurons | 128 |
| Hidden Layers | 3 |
| Dropout Rate | 0.2 |
| Activation Function | Tanh |
| Epochs | 30 |
| Batch Size | 32 |
| Cross-Validation Accuracy | 0.8619 |

### Optimized Model Performance (Test Set)

| Metric | Score |
|--------|-------|
| Accuracy | 0.8790 |
| Precision | 0.8786 |
| Recall | 0.8790 |
| F1-Score | 0.8777 |

### Baseline vs. Optimized Model Comparison

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Accuracy | 0.8802 | 0.8790 |
| Precision | 0.8813 | 0.8786 |
| Recall | 0.8802 | 0.8790 |
| F1-Score | 0.8798 | 0.8777 |

---

## Key Observations

- **Baseline Performance:** The baseline MLP (784 → Dense(128) → Dense(64) → Dense(10)) achieves ~88% accuracy on the Fashion-MNIST test set after 20 epochs, demonstrating that a simple feedforward network can effectively classify fashion items.
- **Challenging Classes:** The `Shirt` class consistently shows the lowest F1-score (0.67), likely due to its visual similarity to T-shirts, Pullovers, and Coats. `Trouser` achieves the highest precision (0.99) as it is visually distinct from all other categories.
- **Hyperparameter Optimization:** The Randomized Search identified RMSProp (lr=0.001), 3 hidden layers with 128 neurons, Tanh activation, and 0.2 dropout as the best configuration. Despite a longer training schedule (30 epochs), the optimized model performs comparably to the baseline, suggesting the baseline architecture is already well-suited for this dataset.
- **Normalization:** Pixel normalization to [0, 1] is critical for stable and fast training of the MLP, as raw pixel values [0, 255] can cause unstable gradients.
- **Dropout Regularization:** A dropout rate of 0.2 in the optimized model helps mitigate overfitting, which is observable from the training vs. validation accuracy gap in the baseline model during later epochs.

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. Haykin, S. (2009). *Neural Networks and Learning Machines*. Pearson.
4. Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*.
5. TensorFlow/Keras Documentation. https://www.tensorflow.org/api_docs

# Additional Exercise: Multi-Layer Perceptron for XOR Classification

## Objective

Implement a Multi-Layer Perceptron (MLP) using TensorFlow/Keras to solve the XOR classification problem and compare its performance with a single-layer perceptron.

## Experimental Procedure

- Implement a single-layer perceptron for the XOR gate.
- Observe that the perceptron fails to converge since XOR is not linearly separable.
- Build an MLP with one hidden layer using TensorFlow/Keras.
- Train the network using the Adam optimizer and binary cross-entropy loss.
- Evaluate the model using prediction accuracy.
- Plot the training accuracy, training loss, and learned decision boundary.


## Observation

The single-layer perceptron failed to solve the XOR problem because the classes are not linearly separable. The Multi-Layer Perceptron successfully learned the nonlinear decision boundary, achieving correct classification of all XOR input combinations. The training accuracy reached 100% while the loss decreased steadily, demonstrating successful convergence.
