# Deep Learning Laboratory (CS3807)

> Shiv Nadar University Chennai | Academic Year 2026–27

This repository contains the programming assignments and experiments conducted for the Deep Learning Laboratory (CS3807) course. Each experiment is organized into its own folder, complete with a dedicated README file, source code, and dataset information.

A key focus of these experiments is building foundational concepts. Therefore, the core machine learning models are implemented from scratch using fundamental libraries, rather than relying on high-level functions like `model.fit()`. This approach ensures a deeper understanding of the underlying mathematics and algorithms.

---

## Experiments

| Experiment No. | Title | Dataset | Status |
|---|-------|---------|--------|
| 1 | [Single Layer Perceptron](./Lab%201%20-%20Single%20Layer%20Perceptron/) | Banknote Authentication (UCI) | Completed |

Additional experiments will be added as the semester progresses.

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
├── .gitignore
└── README.md                            ← Main repository overview
```

*Note: Generated plots and laboratory manuals are kept locally and are excluded from version control (see `.gitignore`).*

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

Each notebook is designed to be self-contained and ready to execute.
