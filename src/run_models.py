def run_logreg_subjects(subject, DB, mode="detection", nseizures_train=3, SOP = 30) -> dict:
    """
    Load features and labels from disk, split into train/test using pseudoprospective splitting,
    scale, select features, train logistic Regression classifier, and return metrics.

    Parameters
    ----------
    subject : str
        Subject ID
    DB : str or Path
        Root path of BIDS derivatives folder
    mode : str
        "detection" or "prediction"
    nseizures_train : int
        Number of seizures to include in training set

    Returns
    -------
    dict : diverse metrics and predictions
    """

    # --- Import packages inside method ---
    import numpy as np
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, average_precision_score, roc_auc_score
    )
    # Make the Bids path to the linear_features derivative at the subject eeg
    sub_path = Path(DB) / "derivatives" / "linear_features" / f"sub-{subject}" / "eeg"
    # --- Load and concatenate all features ---
    X_list = []
    feat_files = list(sub_path.glob("sub-*_linear-features.npz")) # these are in correct order ie run - 00 01 02 03  if user is sane
    for f in feat_files:
        data = np.load(f, allow_pickle=True)
        X_list.append(data["X"])
    X = np.concatenate(X_list, axis=0)  # concatenate along epochs
    print("X.shape", X.shape)
    print("X NaNs:", np.isnan(X).any())
    if np.isnan(X).any():
        print(f"X has NaN, {np.isnan(X).mean() * 100} percentage xnans found")
        print("replace nan values with 0.5 for now")
        X = np.nan_to_num(X, nan=0.5)
    # --- Load all detection label files and concatenate ---

    if mode == "detection":
        y_list = []
        det_files = list(sub_path.glob("sub-*_detection-labels.npy")) #  magic wildcard results in run 00 01 temporal ordered files again, naja......
        for f in det_files:
            y_list.append(np.load(f).astype(np.int8))
        y = np.concatenate(y_list, axis=0)
        print(len(y_list))
        print(y.shape)



    elif mode == "prediction":
         y_list = []
         mask_list = []
         pred_files = list(sub_path.glob(f"sub-*_prediction-labels_SOP-{SOP}*.npz")) # ....
         for f in pred_files:
             data = np.load(f, allow_pickle=True)
             y_list.append(data["y"].astype(np.int8))
             mask_list.append(data["valid"].astype(bool))
         y = np.concatenate(y_list, axis=0)
         valid_mask = np.concatenate(mask_list, axis=0)
         print(valid_mask.shape)
         print(X.shape)
         print(y.shape)
         print("--- Apply valid mask (for prediction) ---")
         # --- Apply valid mask  ---
         X = X[valid_mask]
         y = y[valid_mask]
         print("X dtype:", X.dtype)
         print("Is masked:", np.ma.isMaskedArray(X))
    # here such a situation :  000111000011100 -> 000011110000 can occurr, reduciong the number of test seizures
    # below the minimum 1 thus the try condition
    print("train test split")
    # Scan until we've included N seizures
    # Find all seizure starts
    seizure_starts = np.where((y[:-1] == 0) & (y[1:] == 1))[0] + 1
    print(f"seizure_starts={seizure_starts}")
    if y[0] == 1:  # first epoch is seizure
        seizure_starts = np.insert(seizure_starts, 0, 0)

    # Take the N-th seizure
    try:
        nth_start = seizure_starts[nseizures_train - 1]
    except: # fail gracefully if sub id 4 situation of closeby seizures ie SOP SPH window deletes
        print(
            f"Subject {subject} Failed ! Invalid prediction horizon or occurrence period for test set creation n_start"
        )
        return None
    # Find the end of the N-th seizure (last consecutive 1)
    nth_end = nth_start
    while nth_end < len(y) and y[nth_end] == 1:
        nth_end += 1

    # Training mask: all epochs up to the end of N-th seizure
    train_mask = np.zeros_like(y, dtype=bool)
    train_mask[:nth_end] = True
    test_mask = ~train_mask

    # everything beyond this idx is test set

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print("X_train NaNs:", np.isnan(X_train).any())
    print("X_test NaNs:", np.isnan(X_test).any())

    print("X_test dtype:", X_test.dtype)
    print(f"[{mode}] Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Training data nans??: {np.isnan(X_train).any()}")
    print(f"Test data nans??: {np.isnan(X_test).any()}")
    # --- Scaling ---
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Feature selection ---
    selector = SelectKBest(f_classif, k=100)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # --- Train classifier ---
    clf = LogisticRegression(max_iter=1200, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    fpr = fp / (fp + tn)
    specificity = tn / (tn + fp)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_prob)
    AUC = roc_auc_score(y_test, y_pred)
    fa_per_hour = fp / (len(y_test)/ (3600/5) )  # 5s epochs

    return {
        "subject": subject,
        "mode": mode,
        "classifier": str(clf),
        "confusion_matrix": cm,
        "predictions": y_pred,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "fpr_hour": fa_per_hour,
        "PR-AUC": ap,
        "AUC": AUC
    }
