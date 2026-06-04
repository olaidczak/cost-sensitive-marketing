# Cost-Sensitive Predictive Marketing

## Repository structure
```
├── data/
│   ├── x_test.txt
│   ├── x_train.txt
│   └── y_train.txt
├── src/
│   ├── feature_selection/
│   │   ├── eda.ipynb                        # Exploratory data analysis
│   │   ├── feature_importance_rank.csv      # Ranked feature importances
│   │   ├── feature_rank.py                  # Feature ranking logic
│   │   └── selected_features.txt            # 30 selected features
│   └── model/
│       ├── utils.py                         # Shared utilities
│       ├── run.py                           # Model training/inference entry point
│       ├── cv_results_seed_3105.csv         # Cross-validation results
│       ├── model_selection.ipynb            # Model selection experiments
│       └── tabpfn.ipynb                     # TabPFN experiments
└── requirements.txt

```
