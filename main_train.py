import numpy as np
from feature_extraction.linear_features import univariate_linear_features
from feature_extraction.events_to_annot import get_seizure_intervals, label_epochs, valid_patids
from bids import BIDSLayout
import mne
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import Log_reg_model.utils as utils
epoch_length = 5.0
sph = 10
window_length = 5

# IO setup here
# .. define paths to BIDS DBS containing eeg data in (mne-readable) edf format
siena = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena'
chbmit = '/Volumes/Extreme SSD/EEG_Databases/BIDS_CHB-MIT'

DB = siena # select current bd for testing

# load BIDS layout for parsing and data access
layout = BIDSLayout(DB, validate=False)
# select valid patient population based identified criteria
print(f"valid {valid_patids[siena]}")

## load in the MNE raw eeg signal instance
for subject in valid_patids[DB]:

    print(f"--------------------- Subject {subject} ---------------------- ")

    train_features_list, test_features_list = [],[]
    train_label_list, test_label_list = [],[]

    all_edf_files = layout.get(subject=subject, extension='edf', return_type="file")
    all_tsv_files = layout.get(subject=subject, extension='tsv', return_type="file")

    # Use first three seizures as training
    # assumes chronologically ordered run files: 00 < 01 < 02 etc

    train_edf_files,test_edf_files = all_edf_files[0:3], all_edf_files[3:]
    train_tsv_files,test_tsv_files = all_tsv_files[0:3], all_tsv_files[3:]


    for edf_file,tsv_file in zip(all_edf_files,all_tsv_files):
        # read in the edf files and annotations from the events-table
        edf = mne.io.read_raw_edf(edf_file, preload=True).filter(l_freq=1., h_freq=40.,verbose=False)
        events = pd.read_csv(tsv_file, sep="\t")

        # TODO Make a detection vs prediction task here based on sop; sph
        # TODO Catch Datetime obj for compapipity

        annotations = mne.Annotations(
            onset = events["onset"].values,
            duration = events.get("duration", np.zeros(len(events))).values,
            description = events["eventType"].astype(str).values,
        )
        # add anntotion to edf channel
        edf.set_annotations(annotations)
        ## Epoch data
        epochs = mne.make_fixed_length_epochs(
            edf,
            duration=epoch_length,
            overlap=0,
            preload=True,
            verbose=False
        )

        labels = label_epochs(epochs, get_seizure_intervals(events))

        print("Interictal epochs:", np.sum(labels == 0))
        print("Ictal epochs:", np.sum(labels == 1))
        epochs.metadata = pd.DataFrame({"label": labels})
        # add the label vector to all vectors
        # calculate 59 linear univariate features for each channel and epoch in the edf file
        features_df = univariate_linear_features(epochs)

        if edf_file in train_edf_files:
            train_label_list += [labels]
            train_features_list += [features_df]

        elif edf_file in test_edf_files:
            test_label_list += [labels]
            test_features_list += [features_df]

    # patient training features X and labels Y np arrays are the concat of
    # the respective lists across all patients edf files
    X_train,X_test = np.concatenate(train_features_list), np.concatenate(test_features_list)
    y_train,y_test = np.concatenate(train_label_list), np.concatenate(test_label_list)
    print("scaling ..")
    # training features
    scaler = StandardScaler().fit(X_train)
    training_features = scaler.transform(X_train)
    testing_features = scaler.transform(X_test)

    #################### Feature Selection #######################
    feature_selection = SelectKBest(f_classif, k=10)
    training_features = feature_selection.fit_transform(training_features, y_train)
    testing_features = feature_selection.transform(X_test)

    #################### Classification ###########################
    class_weights = utils.computeBalancedClassWeights(y_train)
    sample_weights = utils.computeSampleWeights(y_train, class_weights)

    logreg = LogisticRegression()
    print("Training model...")
    logreg.fit(training_features, y_train, sample_weight=sample_weights)

    classification_labels = []
    print(logreg.score(testing_features, y_test))
    ypred = logreg.predict(testing_features)
    print("Confusion matrix:")
    confusion_matrix = confusion_matrix(y_test, ypred)
    print(confusion_matrix)

    forecast_labels, exact_labels = utils.FiringPower(ypred, sop, window_length, "fore")
    seizure_sensitivityP = utils.seizureSensitivity(ypred, y_test)
   # FPR = utils.falsePositiveRate(ypred, testing_labels, sph + sop, testing_datetimes, testing_onsets,
                                 # window_length)

    print("Sensitivity: " + str(seizure_sensitivityP))
   # print("FPR/h: " + str(FPR))

    if subject is valid_patids[DB][0] :
        break

#epochs["label == 1"]   # ictal
#epochs["label == 0"]   # interictal


