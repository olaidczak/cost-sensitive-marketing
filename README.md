# Cost-Sensitive Predictive Modeling

## Setup
Python 3.13.7 was utilized

1. **Clone the repository**
```bash
   git clone https://github.com/olaidczak/cost-sensitive-marketing.git
```

2. **Create and activate a virtual environment**
```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate         # Windows
```

3. **Install packages**
```bash
   pip install -r requirements.txt
```

## Repository structure

├── data/
│   ├── x_test.txt
│   ├── x_train.txt
│   └── y_train.txt
├── src/
│   ├── feature_selection/
│   │   ├── eda.ipynb                        # exploratory data analysis
│   │   ├── feature_importance_rank.csv      # ranked feature importances
│   │   ├── feature_rank.py                  # feature ranking logic
│   │   └── selected_features.txt            # 30 selected features
│   └── model/
│       ├── utils.py                         # shared utilities for experiments
│       ├── run.py                           # script for random search cv
│       ├── cv_results_seed_3105.csv         # random search cv results
│       ├── *_res.csv                        # results for top20 models 50/50 split
│       ├── model_selection.ipynb            # model selection experiments
│       └── tabpfn.ipynb                     # TabPFN experiments - not utilized
└── requirements.txt

```
