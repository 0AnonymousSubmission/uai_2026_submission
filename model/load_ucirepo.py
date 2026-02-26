import torch
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# NOTE: obesity (544) has categorical target but UCI lists it as Classification/Regression/Clustering.
# We use it as ordinal regression (category codes 0-6).
# NOTE: seoulBike (560) has wrong target in UCI metadata ("Functioning Day" instead of "Rented Bike Count").
# We fix this via DATASETS_WITH_TARGET_FIX.
datasets = [
    ("student_perf", 320, "regression"),
    ("abalone", 1, "regression"),
    ("obesity", 544, "regression"),
    ("bike", 275, "regression"),
    ("realstate", 477, "regression"),
    ("energy_efficiency", 242, "regression"),
    ("concrete", 165, "regression"),
    ("ai4i", 601, "regression"),
    ("appliances", 374, "regression"),
    ("iris", 53, "classification"),
    ("hearth", 45, "classification"),
    ("winequalityc", 186, "classification"),
    ("breast", 17, "classification"),
    ("adult", 2, "classification"),
    ("bank", 222, "classification"),
    ("wine", 109, "classification"),
    ("car_evaluation", 19, "classification"),
    ("student_dropout", 697, "classification"),
    ("mushrooms", 73, "classification"),
    ("seoulBike", 560, "regression"),
]

DATASETS_WITH_TARGET_FIX = {
    560: "Rented Bike Count",
}


def one_hot_with_cap(X, cap=100):
    # Separate numeric and categorical
    num_X = X.select_dtypes(exclude=["object", "category"])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # How many feature slots are available for one-hot columns
    available = cap - num_X.shape[1]
    if available <= 0 or len(cat_cols) == 0:
        # No room for any one-hot features or no categoricals
        out = num_X.copy()
        return out, num_X.columns.tolist(), []

    # Count classes per categorical column
    class_counts = X[cat_cols].nunique(dropna=True)

    # Decide which categorical columns to drop to fit the cap
    total_needed = int(class_counts.sum())
    to_drop = []

    if total_needed > available:
        # Drop largest cardinality columns until we fit
        for col, cnt in class_counts.sort_values(ascending=False).items():
            if total_needed <= available:
                break
            to_drop.append(col)
            total_needed -= int(cnt)

    keep_cols = [c for c in cat_cols if c not in to_drop]

    # One-hot encode kept categoricals
    if keep_cols:
        dummies = pd.get_dummies(X[keep_cols], prefix=keep_cols, dummy_na=True, dtype=int)
        out = pd.concat([num_X, dummies], axis=1)
        dummy_cols = dummies.columns.tolist()
    else:
        out = num_X.copy()
        dummy_cols = []

    # If overflow, trim extra dummy columns
    if out.shape[1] > cap:
        all_num_cols = num_X.columns.tolist()
        room = max(cap - len(all_num_cols), 0)
        trimmed_dummy_cols = dummy_cols[:room]
        out = pd.concat([num_X, out[trimmed_dummy_cols]], axis=1)
        dummy_cols = trimmed_dummy_cols

    return out, num_X.columns.tolist(), dummy_cols


def get_ucidata(dataset_id, task, device="cuda", cap=50):
    dataset = fetch_ucirepo(id=dataset_id)

    X = dataset.data.features
    y = dataset.data.targets

    if dataset_id in DATASETS_WITH_TARGET_FIX:
        target_col = DATASETS_WITH_TARGET_FIX[dataset_id]
        y = X[[target_col]]
        X = X.drop(columns=[target_col])

    X = X.dropna(axis=1)

    # One-hot with bookkeeping of which columns are numeric vs one-hot
    X_all, orig_num_cols, dummy_cols = one_hot_with_cap(X, cap=cap)

    # If y is categorical/string, convert to category codes (ordinal encoding)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if (
        pd.api.types.is_string_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
        or y.dtype == "object"
    ):
        y = y.astype("category").cat.codes.astype(float)

    # Baseline the categories to be 0, 1, ..., num_classes-1
    if task == "classification":
        class_dict = {old: new for new, old in enumerate(sorted(y.unique()))}
        y = y.map(class_dict)

    # Split into train, val, test on the already encoded features
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_all, y, test_size=0.3, random_state=42
    )
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=0.5, random_state=42
    )

    # Fit StandardScaler on training numeric columns only
    scaler = StandardScaler()  #
    # Some datasets may have zero numeric columns after preprocessing
    if len(orig_num_cols) > 0:
        scaler.fit(X_train_df[orig_num_cols])

        # Transform train, val, test numeric columns in place, leave one-hot columns unchanged
        # Set the original columns to floats
        X_train_df[orig_num_cols] = X_train_df[orig_num_cols].astype(float)
        X_val_df[orig_num_cols] = X_val_df[orig_num_cols].astype(float)
        X_test_df[orig_num_cols] = X_test_df[orig_num_cols].astype(float)

        X_train_df.loc[:, orig_num_cols] = scaler.transform(X_train_df.loc[:, orig_num_cols])
        X_val_df.loc[:, orig_num_cols] = scaler.transform(X_val_df.loc[:, orig_num_cols])
        X_test_df.loc[:, orig_num_cols] = scaler.transform(X_test_df.loc[:, orig_num_cols])

    # Convert to tensors
    X_train = torch.tensor(X_train_df.values, dtype=torch.float64, device=device)
    y_train = torch.tensor(
        y_train.values, dtype=torch.float64 if task == "regression" else torch.long, device=device
    )

    X_val = torch.tensor(X_val_df.values, dtype=torch.float64, device=device)
    y_val = torch.tensor(
        y_val.values, dtype=torch.float64 if task == "regression" else torch.long, device=device
    )

    X_test = torch.tensor(X_test_df.values, dtype=torch.float64, device=device)
    y_test = torch.tensor(
        y_test.values, dtype=torch.float64 if task == "regression" else torch.long, device=device
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
