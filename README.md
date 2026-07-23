# Deep Learning Laboratory (CS3807)

> Shiv Nadar University Chennai | Academic Year 2026–27

This repository contains the programming assignments and experiments conducted for the Deep Learning Laboratory (CS3807) course. Each experiment is organized into its own folder, complete with a dedicated README file, source code, and dataset information.


---

## Experiments

| Experiment No. | Title | Dataset | Status |
|---|-------|---------|--------|
| 1 | [Single Layer Perceptron](./Lab%201%20-%20Single%20Layer%20Perceptron/) | Banknote Authentication (UCI) | Completed |
| 2 | [Multi-Layer Perceptron (MLP) for Image Classification](./Lab%202%20-%20Multi-Layer%20Perceptron/) | Fashion-MNIST | Completed |



---

## Repository Structure

```
DL Lab/
├── Lab 1 - Single Layer Perceptron/
│   ├── README.md                        ← Experiment details and theory
│   ├── DL_Lab_1.ipynb                   ← Implementation source code
│   ├── requirements.txt                 ← Required libraries
│   └── banknote+authentication/
│       └── data_banknote_authentication.txt
├── Lab 2 - Multi-Layer Perceptron/
│   ├── README.md                        ← Experiment details and theory
│   ├── DL_Lab_2.ipynb                   ← Implementation source code
│   ├── Experiment_2_Lab_Manual.pdf      ← Lab manual
│   └── *.eps                            ← Generated output plots
├── .gitignore
└── README.md                            ← Main repository overview
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

