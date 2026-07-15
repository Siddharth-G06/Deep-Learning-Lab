# Experiment 1 — Single Layer Perceptron

**Course:** CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

---

## What this is about

The goal here is to build a perceptron from the ground up and see how it actually learns — not just call `sklearn.Perceptron()` and move on. The dataset is the **Banknote Authentication** dataset from UCI, where the task is to figure out whether a given banknote is genuine or forged based on four wavelet-transform features.

It's a classic binary classification problem and a natural starting point for understanding how a single artificial neuron works: how weights update, what the step activation does, why the learning rate matters, and where the perceptron breaks down.

---

## Dataset

**Banknote Authentication Dataset** — [UCI ML Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

The dataset has 1,372 samples and 4 features extracted from images of banknotes using Wavelet Transform:

| Feature | Description |
|---------|-------------|
| `variance` | Variance of wavelet-transformed image |
| `skewness` | Skewness of wavelet-transformed image |
| `curtosis` | Curtosis of wavelet-transformed image |
| `entropy` | Entropy of the image |
| `class` | 0 = Authentic, 1 = Forged |

- **762** authentic samples, **610** forged
- No missing values
- File: `banknote+authentication/data_banknote_authentication.txt` (no header row — column names assigned in code)

---

## What the notebook covers

The notebook is structured around tasks:

1. **Dataset exploration** — load, check shape, inspect missing values, class distribution
2. **EDA** — histograms, correlation heatmap, scatter/pair plot, box plots
3. **Preprocessing** — MinMax normalisation + 80/20 stratified split
4. **Perceptron from scratch** — a clean `Perceptron` class with:
   - step activation function
   - weight + bias initialisation to zero
   - perceptron learning rule: `w = w + lr * (y - ŷ) * x`
   - per-epoch error tracking, weight history, bias history
5. **Training** — run for 100 epochs at `lr = 0.01`
6. **Evaluation** — accuracy, precision, recall, F1, confusion matrix
7. **Learning rate comparison** — same model at `lr = 0.001, 0.01, 0.1`, convergence plot

---

## Folder structure

```
Lab 1 - Single Layer Perceptron/
├── README.md
├── DL_Lab_1.ipynb
├── requirements.txt
└── banknote+authentication/
    └── data_banknote_authentication.txt
```

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Running it

```bash
cd "Lab 1 - Single Layer Perceptron"
pip install -r requirements.txt
jupyter notebook DL_Lab_1.ipynb
```

Run cells top to bottom. Plots get saved as `.eps` files locally (not tracked in git).

---

## Key observations

- The perceptron converges cleanly on this dataset since the classes are nearly linearly separable in the transformed feature space.
- Lower learning rates (`0.001`) converge more slowly but more steadily. Higher rates (`0.1`) can overshoot.
- MinMax normalisation makes a noticeable difference in how quickly the weights stabilise.
- Weight and bias evolution plots make it easy to see *when* the model stops meaningfully updating.
