# Deep Learning Laboratory (CS3807)

> Shiv Nadar University Chennai | Academic Year 2026-27

This repository contains the programming assignments and experiments conducted for the Deep Learning Laboratory (CS3807) course. Each experiment is organized into its own folder, complete with a dedicated README file, source code, and dataset information.

---

## Experiments

| Experiment No. | Title | Dataset | Status |
|---|-------|---------|--------|
| 1 | [Single Layer Perceptron](./Lab%201%20-%20Single%20Layer%20Perceptron/) | Banknote Authentication (UCI) | Completed |
| 2 | [Multi-Layer Perceptron (MLP) for Image Classification](./Lab%202%20-%20Multi-Layer%20Perceptron/) | Fashion-MNIST | Completed |
| 3 | [Convolutional Neural Network (CNN) for Image Classification](./Lab%203%20-%20Convolutional%20Neural%20Network/) | CIFAR-10 | Completed |
| 4 | [Comparative Study of Deep CNN Architectures Using Transfer Learning](./Lab%204-%20Deep%20CNN%20Architecture%20Using%20Transfer%20Learning/) | CIFAR-10 | Completed |
| 5 | [Comprehensive Study of CNN Training](./Lab%205%20-%20Comprehensive%20Study%20of%20CNN%20Training/) | Oxford-IIIT Pet Dataset | Completed |
| 6 | [End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding](./Lab%206%20-%20End-to-End%20Study%20of%20RNN%2C%20LSTM%20and%20GRU/) | UCI HAR + UCF101 (video) | Completed |

---

## Repository Structure

```text
DL Lab/
+-- Lab 1 - Single Layer Perceptron/
|   +-- README.md                        # Experiment details and theory
|   +-- DL_Lab_1.ipynb                   # Implementation source code
|   +-- requirements.txt                 # Required libraries
|   +-- banknote+authentication/
|       +-- data_banknote_authentication.txt
+-- Lab 2 - Multi-Layer Perceptron/
|   +-- README.md                        # Experiment details and theory
|   +-- DL_Lab_2.ipynb                   # Implementation source code
|   +-- Experiment_2_Lab_Manual.pdf      # Lab manual
|   +-- *.eps                            # Generated output plots
+-- Lab 3 - Convolutional Neural Network/
|   +-- README.md                        # Experiment details and theory
|   +-- DL_Lab_3.ipynb                   # Implementation source code
|   +-- sample_images.eps                # Generated output plots
+-- Lab 4- Deep CNN Architecture Using Transfer Learning/
|   +-- README.md                        # Experiment details and theory
|   +-- DL_Lab_4.ipynb                   # Implementation source code
|   +-- Experiment_4_Lab_Manual.pdf      # Lab manual
|   +-- requirements.txt                 # Required libraries
|   +-- *.eps                            # Generated output plots
+-- Lab 5 - Comprehensive Study of CNN Training/
|   +-- README.md                        # Experiment details and theory
|   +-- dl-lab-5.ipynb                   # Implementation source code
|   +-- images/                          # Images folder for the lab
+-- Lab 6 - End-to-End Study of RNN, LSTM and GRU/
|   +-- README.md                        # Experiment details and theory
|   +-- Experiment6_Complete_Lab.ipynb   # Implementation source code
|   +-- Experiment_6.pdf                 # Lab manual (PDF)
|   +-- Experiment_6(1).tex              # Lab manual (LaTeX source)
|   +-- plot*.png                        # Generated output plots
+-- .gitignore
+-- README.md                            # Main repository overview
```

---

## Execution Instructions

To run any of the experiments locally, please follow these steps:

1. Clone the repository and navigate to the specific experiment folder.
2. Install the necessary dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook to view and execute the code:
   ```bash
   jupyter notebook
   ```