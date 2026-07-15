# Deep Learning Lab — CS3807

> Shiv Nadar University Chennai · AY 2026–27

This is where I keep all the code I write for my Deep Learning lab (CS3807). Each experiment lives in its own folder with its own README, so you can jump straight to whatever you're looking for without wading through a single massive notebook.

The implementations are done **from scratch** — no black-box calls to `model.fit()` and done. The whole point is to actually understand what's happening under the hood, and that's what this repo tries to reflect.

---

## Experiments

| # | Title | Dataset | Status |
|---|-------|---------|--------|
| 1 | [Single Layer Perceptron](./Lab%201%20-%20Single%20Layer%20Perceptron/) | Banknote Authentication (UCI) | ✅ Done |

More experiments will be added as the semester goes on.

---

## How this repo is organised

```
DL Lab/
├── Lab 1 - Single Layer Perceptron/
│   ├── README.md                        ← what the experiment is about
│   ├── DL_Lab_1.ipynb                   ← the actual code
│   ├── requirements.txt
│   └── banknote+authentication/
│       └── data_banknote_authentication.txt
├── .gitignore
└── README.md                            ← you're here
```

Plots are generated inside the notebook and saved locally — they're not tracked in git (see `.gitignore`). Lab manuals are also excluded for the same reason.

---

## Running any experiment

1. Clone the repo and `cd` into the relevant experiment folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Fire up Jupyter and open the notebook:
   ```bash
   jupyter notebook
   ```

That's it, no special setup needed. Each notebook is self-contained.

---

*Built as part of coursework — but written to actually make sense.*
