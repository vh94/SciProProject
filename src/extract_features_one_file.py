# This script showcases the univariate_linear_features method
# from the linear_features submodule.
# In the example it will be used to calculate the 59 univariate features as described in ... and later used by ...
# for all channels of a singular EDF file.

import mne
import linear_features
import pandas as pd
import time
import os
from feature_extraction.events_to_annot import labels_for_prediction, labels_for_detection
print(f"mne version: {mne.__version__}")

# Load the EDF file
edf_file = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena/sub-00/ses-01/eeg/sub-00_ses-01_task-szMonitoring_run-02_eeg.edf'  # Replace with the path to your EDF file
raw = mne.io.read_raw_edf(edf_file, preload=True)
#mne.viz.plot_raw(raw)

# load annotations
ann_file = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena/sub-00/ses-01/eeg/sub-00_ses-01_task-szMonitoring_run-02_events.tsv'
annot = pd.read_csv(ann_file, sep='\t')

annot
#getLabelsForSeizure(annot.dateTime.values, 0, annot.onset.values) # ????

# Apply a bandpass filter (optional)
raw.filter(l_freq=1, h_freq=40)

# Define the window length (5 seconds) and create epochs
epoch_length = 5  # 5 seconds
epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, overlap=0, preload=True)

print(sum(label_epochs)*5)
annot

labels_det =labels_for_detection(epochs, annot)
labels_pred, valid =labels_for_prediction(epochs, annot)

all(label_epochs_f == label_epochs)
#### USING NEW METHOD
start_time = time.time()  # Get the start time
features_df = linear_features.univariate_linear_features(epochs)
print(f"Execution time new: {time.time() - start_time} seconds")

### Check results
print(f"Final features shape: {features_df.shape}")
assert features_df.shape == (epochs.get_data().shape[0],59 * epochs.get_data().shape[1] )

## saving features to numpy array as done in previous studies:::
np.save( f"{os.path.basename(edf_file)}_features.npy",features_df.to_numpy() )
