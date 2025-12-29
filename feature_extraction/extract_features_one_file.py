import mne
import linear_features
from Log_reg_model.utils import *
import pandas as pd
import time
import os

print(f"mne version: {mne.__version__}")

# Load the EDF file
edf_file = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena/sub-00/ses-01/eeg/sub-00_ses-01_task-szMonitoring_run-00_eeg.edf'  # Replace with the path to your EDF file
raw = mne.io.read_raw_edf(edf_file, preload=True)

# load annotations
ann_file = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena/sub-00/ses-01/eeg/sub-00_ses-01_task-szMonitoring_run-00_events.tsv'
annot = pd.read_csv(ann_file, sep='\t')

#getLabelsForSeizure(annot.dateTime.values, 0, annot.onset.values) # ????

# Apply a bandpass filter (optional)
raw.filter(l_freq=1, h_freq=40)

# Define the window length (5 seconds) and create epochs
epoch_length = 5  # 5 seconds
epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, overlap=0, preload=True)

#### USING NEW METHOD
start_time = time.time()  # Get the start time
features_df = linear_features.univariate_linear_features(epochs)
print(f"Execution time new: {time.time() - start_time} seconds")

### Check results
print(f"Final features shape: {features_df.shape}")
assert features_df.shape == (epochs.get_data().shape[0],59 * epochs.get_data().shape[1] )

## saving features to numpy array as done in previous studies:::
np.save( f"{os.path.basename(edf_file)}_features.npy",features_df.to_numpy() )



#scaler = StandardScaler().fit(features_df)


 ## Train test split

 # train == first five features.....
#training_features = scaler.transform(features_df)


 #
 ##################### Data Sampling ###########################
 ## no data sampling -> sample weight
 #
 ##################### Feature Selection #######################
 ## Filter selection with ANOVA-F
 #n_features = k_features[kk]
 #feature_selection = SelectKBest(f_classif, k=n_features)
 #training_features = feature_selection.fit_transform(training_features,
                                                     #training_labels)
 #validation_features = feature_selection.transform(validation_features)
 #
 ##################### Classification ###########################
 #
 #class_weights = utils.computeBalancedClassWeights(training_labels)
 #sample_weights = utils.computeSampleWeights(training_labels, class_weights)
 #
 #logreg = LogisticRegression()
 #logreg.fit(training_features, training_labels, sample_weight=sample_weights)
 #
 ####################### Performance Evaluation #########################
 #predicted_labels = logreg.predict(validation_features)
 #tn, fp, fn, tp = confusion_matrix(validation_labels, predicted_labels).ravel()
 #
 #performance = np.sqrt(utils.specificity(tn, fp) * utils.sensitivity(tp, fn))
 #
 #performance_values[i, kk] = performance_values[i, kk] + performance
