
def create_features_labels_single_subject(subject, DB):
    import numpy as np
    import pandas as pd
    import mne
    from bids import BIDSLayout
    from pathlib import Path
    from feature_extraction.linear_features import univariate_linear_features, linear_feature_names
    from feature_extraction.events_to_annot import (
        labels_for_detection, labels_for_prediction
    )

    print(f"--------------------- Subject {subject} ----------------------")

    # Recreate layout INSIDE process
    layout = BIDSLayout(DB, validate=False)
    all_edf_files = layout.get(subject=subject, extension='edf', return_type="file")
    all_tsv_files = layout.get(subject=subject, extension='tsv', return_type="file")

    # Make an identical dir to store the derivatives:::
    deriv_root = Path(DB) / "derivatives" / "linear_features"
    sub_path = deriv_root / f"sub-{subject}" / "eeg"
    sub_path.mkdir(parents=True, exist_ok=True)

    for edf_file, tsv_file in zip(all_edf_files, all_tsv_files):
        #### ERRRROR CONCAT MISSING
        edf_file = Path(edf_file)
        edf = (
            mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            .filter(1., 40., verbose=False)
        )
        events = pd.read_csv(tsv_file, sep="\t")
        print(f"{edf_file} has {len(events)} events")
        print(events)
        #annotations = mne.Annotations(
            #onset=events["onset"].values,
            #duration=events.get("duration", np.zeros(len(events))).values,
            #description=events["eventType"].astype(str).values,
        #)
        #edf.set_annotations(annotations)
        epochs = mne.make_fixed_length_epochs(
            edf, duration=5.0, preload=True, verbose=False
        )
        print("Calculating 59 univariate linear Features..")
        features_df = univariate_linear_features(epochs)
        # convert back to numpy.....
        X = features_df.to_numpy(dtype=np.float32)
        cols = features_df.columns.to_numpy()
        # Wirte the features to BIDS database derivatives linear_features
        feat_path = sub_path / f"{edf_file.stem}_linear-features.npz"
        np.savez(
            feat_path,
            X=X,
            cols=cols
        )
        ########  LABELLING PART
        #ch_names = layout.get_channel_names(subject=subject, datatype="eeg")
        print("Labeling epochs")
        labels_detection = labels_for_detection(epochs,events)
        # detection labels (.npy)
        det_path = sub_path / f"{edf_file.stem}_detection-labels.npy"
        np.save(det_path, labels_detection.astype(np.int8))

        # labels for prediciton SOP values in minutes see costa et al...
        sop_minutes = range(20, 55, 5)  # 20, 25, ..., 50
        SPH = 10 * 60  # seconds
        for sop_min in sop_minutes:
            SOP = sop_min * 60  # seconds

            print(f"Labeling prediction epochs | SOP={sop_min}min SPH=10min")

            labels_prediction, valid_tp = labels_for_prediction(
                epochs,
                events,
                SOP=SOP,
                SPH=SPH,
            )

            pred_path = sub_path / (
                f"{edf_file.stem}_prediction-labels_"
                f"SOP-{sop_min:02d}min_SPH-10min.npz"
            )

            np.savez(
                pred_path,
                y=labels_prediction.astype(np.int8),
                valid=valid_tp.astype(bool),
                SOP=SOP,
                SPH=SPH,
            )
    return None


def train_evaluate_subject_from_disk(subject, DB, mode="detection", nseizures_train=3):
    """
    Load features and labels from disk, split into train/test,
    scale, select features, train classifier, and return metrics.

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
    dict : metrics and predictions
    """

    # --- Import packages inside method ---
    import numpy as np
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, average_precision_score
    )
    sub_path = Path(DB) / "derivatives" / "linear_features" / f"sub-{subject}" / "eeg"
    # --- Load and concatenate all features ---
    X_list = []
    feat_files = list(sub_path.glob("sub-*_linear-features.npz")) # are these in correct order???
    print(feat_files)
    for f in feat_files:
        data = np.load(f, allow_pickle=True)
        X_list.append(data["X"])
    X = np.concatenate(X_list, axis=0)  # concatenate along epochs

    # --- Load all detection label files and concatenate ---

    if mode == "detection":
        y_list = []
        det_files = list(sub_path.glob("sub-*_detection-labels.npy"))          # again are these in correct order
        print(det_files)
        for f in det_files:
            y_list.append(np.load(f).astype(np.int8))
        y = np.concatenate(y_list, axis=0)
    print(len(y_list))
     #elif mode == "prediction":
         #y_list = []
         #pred_files = list(sub_path.glob("sub-*_prediction-labels_*.npz"))
         #for f in pred_files:
             #y_list.append(np.load(f).astype(np.int8))
         #y = np.concatenate(y_list, axis=0)
         #data = np.load(pred_file, allow_pickle=True)
         #y = data["y"].astype(np.int8)
         #valid_mask = data["valid"].astype(bool)
         ## --- Apply valid mask  ---
         #X = X_all[valid_mask]
         #y = y[valid_mask]
    print( "--- Apply valid mask (for prediction) ---")

    print("train test split")
    # Scan until we've included N seizures
    # Find all seizure starts
    seizure_starts = np.where((y[:-1] == 0) & (y[1:] == 1))[0] + 1
    print(f"seizure_starts={seizure_starts}")
    if y[0] == 1:  # first epoch is seizure
        seizure_starts = np.insert(seizure_starts, 0, 0)

    # Take the N-th seizure
    nth_start = seizure_starts[nseizures_train - 1]

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

    print(f"[{mode}] Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # --- Scaling ---
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Feature selection ---
    selector = SelectKBest(f_classif, k=40)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # --- Train classifier ---
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
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
    fa_per_hour = fp / (len(y_test) * 5 / 3600)  # 5s epochs

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
    }

### Test
DB = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena'
#
sub = "06"
import time

start = time.time()
create_features_labels_single_subject(sub,DB)
out = train_evaluate_subject_from_disk(sub,DB)
# seizure_starts=[ 228  769 1138 1687 2086]
# [detection] Training data shape: (1150, 1121), Test data shape: (1184, 1121)out
end = time.time()


print(f"Execution time: {end - start:.4f} seconds")

with open("log.txt", "a") as f:
    f.write(f" {int(time.time())}   {'-' * 60}\n")
    f.write(f"{DB}\n")
    for k, v in out.items():
        f.write(f"{k}: {v}\n")
#
#print("FINISHED ---------------------------------- \n ---------- \n ------------ \n ------------")
#print(out)
# THis function seems to perform slightly better???
# investigate this!
def run_single_subject(subject, DB, nseizures_train = 3):
    import numpy as np
    import pandas as pd
    import mne
    from bids import BIDSLayout
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, confusion_matrix, average_precision_score
    )
    from feature_extraction.linear_features import univariate_linear_features, linear_feature_names
    from feature_extraction.events_to_annot import (
        labels_for_detection, labels_for_prediction
    )

    print(f"--------------------- Subject {subject} ----------------------")

    # Recreate layout INSIDE process
    layout = BIDSLayout(DB, validate=False)


    all_edf_files = layout.get(subject=subject, extension='edf', return_type="file")
    all_tsv_files = layout.get(subject=subject, extension='tsv', return_type="file")

    # Make an identical dir to store the derivatives:::
    deriv_root = Path(DB) / "derivatives" / "linear_features"
    sub_path = deriv_root / f"sub-{subject}" / "eeg"
    sub_path.mkdir(parents=True, exist_ok=True)

    # It is not that easy because the test files have to contain at least one seizure !
    #train_edf_files = all_edf_files[:3] # this condition len > 3 has to be checked prior to passing !
    #test_edf_files  = all_edf_files[3:]
    # I count the number of events instead;
    count = 0

    train_features_list, test_features_list = [], []
    train_label_list, test_label_list = [], []

    for edf_file, tsv_file in zip(all_edf_files, all_tsv_files):
        edf = (
            mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            .filter(1., 40., verbose=False)
        )
        events = pd.read_csv(tsv_file, sep="\t")
        print(f"{edf_file} has {len(events)} events")
        print(events) # if theres no events

        annotations = mne.Annotations(
            onset=events["onset"].values,
            duration=events.get("duration", np.zeros(len(events))).values,
            description=events["eventType"].astype(str).values,
        )

        edf.set_annotations(annotations)

        epochs = mne.make_fixed_length_epochs(
            edf, duration=5.0, preload=True, verbose=False
        )

        print("Calculating 59 univariate linear Features..")
        features_df = univariate_linear_features(epochs)
        # convert back to numpy.....
        X = features_df.to_numpy(dtype=np.float32)
        cols = features_df.columns.to_numpy()

        # Wirte the features to BIDS database derivatives linear_features
        # save as .npz
        feat_path = sub_path / f"sub-{subject}_{tsv_file}_linear-features.npz"
        np.savez(
            feat_path,
            X=X,
            cols=cols
        )        ########     LABELLING PART
        #ch_names = layout.get_channel_names(subject=subject, datatype="eeg")
        print("Labeling epochs")
        labels_detection = labels_for_detection(epochs,annotations)
        # detection labels (.npy)
        det_path = sub_path / f"sub-{subject}_{tsv_file}_detection-labels.npy"
        np.save(det_path, labels_detection.astype(np.int8))

        # labels for prediciton SOP values in minutes see costa et al...
        sop_minutes = range(20, 55, 5)  # 20, 25, ..., 50
        SPH = 10 * 60  # seconds

        for sop_min in sop_minutes:
            SOP = sop_min * 60  # seconds

            print(f"Labeling prediction epochs | SOP={sop_min}min SPH=10min")

            labels_prediction, valid_tp = labels_for_prediction(
                epochs,
                annotations,
                SOP=SOP,
                SPH=SPH,
            )

            pred_path = sub_path / (
                f"sub-{subject}_{tsv_file}_prediction-labels_"
                f"SOP-{sop_min:02d}min_SPH-10min.npz"
            )

            np.savez(
                pred_path,
                y=labels_prediction.astype(np.int8),
                valid=valid_tp.astype(bool),
                SOP=SOP,
                SPH=SPH,
            )
        #epochs.metadata = pd.DataFrame({"label": labels_detection}) #
        #epochs.metadata.to_csv(sub_path / f"sub-{subject}_detection-labels.tsv",)


        ### TRAIN TEST SPLIT: Assumes chronological order files
        ### This has to be adapted for the different tasks ie detection, and prediction
        ### ie different labels
        ### for pseudo- projective study, i.e. add first N=3 seizures for training
        if np.sum(labels_detection) > 0: #
            count += 1
            print("Seizure event" ,count)
        if count <= nseizures_train:
            train_features_list.append(features_df)
            train_label_list.append(labels_detection)
            print("Add to training set")
        else:
            test_features_list.append(features_df)
            test_label_list.append(labels_detection)
            print("Add to test set")

    print(len(test_label_list), len(test_features_list))
    ####### SPLIT FUNCTION HERE!!

    ######## TRAIN TEST SPLIT; CLF PART
    if len(test_features_list) == 0:
        print("No test seizures found; ")
        return -1

    X_train = np.concatenate(train_features_list)
    y_train = np.concatenate(train_label_list)
    X_test  = np.concatenate(test_features_list)
    y_test  = np.concatenate(test_label_list)
    print("Saving features")
    np.save("X_train.npy",X_train)
    np.save("y_train.npy",y_train)
    np.save("X_test.npy",X_test)
    np.save("y_test.npy",y_test)


    print(f'Training data shape {X_train.shape}\nTest data shape {X_test.shape}')
    # TODO check shapes !! (nfeatures x nchannels , nwindows)
    #n_feat = 59
    #n_channels =
    #n_windows =

    # TODO write features | y for train and test data to disk!
    # SCALING
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    ########################################### We´re now ready for estimation ...
    # ~ FEATURE SELECTION
    #
    selector = SelectKBest(f_classif, k=40)
    # or varthreshold based
    # var_threshold = np.var(X_train, axis=0).mean()
    # selector = VarianceThreshold(threshold=var_threshold)  # TODO  where is the threshold
    X_train = selector.fit_transform(X_train, y_train)
    X_test  = selector.transform(X_test)
    # SAMPLE WEIGHTS
    # class_weights = utils.computeBalancedClassWeights(y_train)
    # sample_weights = utils.computeSampleWeights(y_train, class_weights) # NOTE This is not a safe function (zeros edgecase)
    # Thus i use the class_weigth arg in the LogisticRegressor model setup::

    ##### Classifier LIST
    #start = time.time()  # record start time

    clfs = [LogisticRegression(max_iter=500, class_weight="balanced")]
    for clf in clfs :
        print(f'Training {clf}')
        #clf.fit(X_train, y_train,sample_weight=sample_weights)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        #sensitivity = utils.seizureSensitivity(y_pred, y_test)
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn)
        specificity = tn / (tn + fp)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_prob)
        fa_per_hour = fp / (len(y_test) * 5 / 3600)

    # TODO Store trained model to disk!

    # TODO analyze performance in detection prediction and forecasting
    return {
        "subject": subject,
        "classifier": f'{clf}',
        "confusion_matrix": cm,
        "predictions": y_pred,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "fpr_hour": fa_per_hour,
        "PR-AUC": ap,
    }

start = time.time()
out = run_single_subject(sub,DB)
end = time.time()


print(f"Execution time: {end - start:.4f} seconds")

with open("log.txt", "a") as f:
    f.write(f" {int(time.time())}   {'-' * 60}\n")
    f.write(f"{DB}\n")
    for k, v in out.items():
        f.write(f"{k}: {v}\n")