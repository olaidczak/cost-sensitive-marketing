import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils import perform_random_search_cv
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

SEED = 3105
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# numbers of features to test
K_VALUES = [2,3,4,5,6,7]
# number of random feature subsets
N_RANDOM = 3
# number of random hyperparameter configs per each model
N_ITER = 20
# fraction of a set we are allowed to contact
CONTACT_RATE = 0.2

data_dir = Path("../../data")
X = pd.read_csv(data_dir / "x_train.txt", sep=" ")
y = pd.read_csv(data_dir / "y_train.txt", sep=" ").values.ravel()

with open("../feature_selection/selected_features.txt") as f:
    selected = [s.strip().strip("'").strip('"') for s in f.read().split(",")]

X = X[selected]

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED
)

X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

print(f"train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
ranking = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(
    ascending=False
)
top15 = ranking.index[:15].tolist()


def random_subsets(k, n, exclude):
    seen = {frozenset(exclude)}
    subsets = []
    while len(subsets) < n:
        s = frozenset(rng.choice(top15, size=k, replace=False))
        if s not in seen:
            seen.add(s)
            subsets.append(sorted(s, key=top15.index))
    return subsets


subsets = {}
for k in K_VALUES:
    topk = ranking.index[:k].tolist()
    topk_1 = ranking.index[1 : k + 1].tolist()
    topk_2 = ranking.index[2 : k + 2].tolist()
    subsets[k] = (
        [("top", topk)]
        + [("top+1", topk_1)]
        + [("top+2", topk_2)]
        + [(f"rand_{i}", s) for i, s in enumerate(random_subsets(k, N_RANDOM, topk))]
    )
    # print(f"k={k}: {len(subsets[k])} subsets")

results = perform_random_search_cv(
    X_train, y_train, seed=SEED, k_values=K_VALUES, n_iter=N_ITER, subsets=subsets
)    