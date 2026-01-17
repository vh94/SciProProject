from scipy.ndimage import sum_labels


def run_single_subject(subject, DB, nseizures_train = 3):
    import numpy as np
    import pandas as pd
    import mne
    from bids import BIDSLayout
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, confusion_matrix
    )
    import Log_reg_model.utils as utils
    from feature_extraction.linear_features import univariate_linear_features
    from feature_extraction.events_to_annot import (
        get_seizure_intervals, label_epochs, label_epochs_new
    )

    print(f"--------------------- Subject {subject} ----------------------")

    # Recreate layout INSIDE process
    layout = BIDSLayout(DB, validate=False)

    train_features_list, test_features_list = [], []
    train_label_list, test_label_list = [], []

    all_edf_files = layout.get(subject=subject, extension='edf', return_type="file")
    all_tsv_files = layout.get(subject=subject, extension='tsv', return_type="file")


    # It is not that easy because the test files have to contain at least one seizure !
    #train_edf_files = all_edf_files[:3] # this condition len > 3 has to be checked prior to passing !
    #test_edf_files  = all_edf_files[3:]
    # I count the number of events instead;
    count = 0
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


        print("Labeling epochs")
        #labels = mne.events_from_annotations(edf)
        #labels = label_epochs(epochs, get_seizure_intervals(events))
        labels = label_epochs_new(epochs,edf.info["sfreq"],annotations)
        print(labels, sum(labels))
        epochs.metadata = pd.DataFrame({"label": labels}) # TODO is the Effective window size : 1.000 (s) coming from here???
        # just create other prediction problem type etc by shifting this labels column ie label_detect , label_predict

        print("Calculating 59 univariate linear Features..")
        features_df = univariate_linear_features(epochs)
        ### TRAIN TEST SPLIT: Assumes chronological order files
        ### for pseudoprojective study, ie add firstn seizures for training
        if np.sum(labels) > 0: # THIS CHECK DOESNT WORK because sum is always zero
            count += 1
            print("Seizure event" ,count)
        if count <= nseizures_train:
            train_features_list.append(features_df)
            train_label_list.append(labels)
            print("Add to training set")
        else:
            test_features_list.append(features_df)
            test_label_list.append(labels)
            print("Add to test set")

    print(len(test_label_list), len(test_features_list))

    if len(test_features_list) == 0:
        print("No test seizures found; ")
        return -1
    X_train = np.concatenate(train_features_list)
    y_train = np.concatenate(train_label_list)
    X_test  = np.concatenate(test_features_list)
    y_test  = np.concatenate(test_label_list)


    print(f'Training data shape {X_train.shape}\nTest data shape {X_test.shape}')
    # TODO check shapes !! (nfeatures x nchannels , nwindows)

    # TODO write features | y for train and test data to disk!
    # SCALING
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    ########################################### We´re now ready for estimation ...
    # ~ FEATURE SELECTION
    #
    selector = SelectKBest(f_classif, k=10)
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
    clfs = [LogisticRegression(max_iter=1000, class_weight="balanced")]
    for clf in clfs :
        print(f'Training {clf}')
        #clf.fit(X_train, y_train,sample_weight=sample_weights)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        #sensitivity = utils.seizureSensitivity(y_pred, y_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

    # TODO Store trained model to disk!

    # TODO analyze perfomance in detection prediction and forecasting
    # And write resuts to logfile
    #print(f"Subject {subject} sensitivity: {sensitivity}")
    sensitivity = 0.0
    return {
        "subject": subject,
        "confusion_matrix": cm,
        "sensitivity": sensitivity,
        "predictions": y_pred,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    }
siena = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena'
from feature_extraction.events_to_annot import valid_patids
sub = subjects = valid_patids[siena][1]
out = run_single_subject(sub,siena)
print("FINISHED ---------------------------------- \n ---------- \n ------------ \n ------------")
print(out)