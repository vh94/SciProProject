from bids import BIDSLayout
import pandas as pd
import mne
import pandas as pd
import numpy as np
import os

# This script is generally intendet to be run once to assign variables
# The minuimum number of seizures to include the subject should be 4 ( 3 train and at least one test)
min_n_sz = 4

# This Dict stores the valid patient ids to be included:
# Define the BIDS DB root paths as the keys in this dictonary::
valid_patids = {
    "/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena"  : [],
    "/Volumes/Extreme SSD/EEG_Databases/BIDS_CHB-MIT": []
}

# Parse all events for a subject to check if more than 4 seizures (3 trainig + 1 test) are avalaible and if
# the time differences are suffienct to use for the study

for bids_root in valid_patids.keys():

    layout = BIDSLayout(bids_root, validate=False)
    valid_subject_ids_ = []

    for subject in layout.get_subjects():

        event_files = layout.get(
            subject=subject,
            suffix="events",
            extension="tsv",
            return_type="file"
        )

        seizure_events = []

        for ef in event_files:
            df = pd.read_csv(ef, sep="\t")

            if "eventType" not in df.columns:
                continue

            seizure_rows = df[df["eventType"].str.lower().str.contains("sz")]

            for idx, row in seizure_rows.iterrows():
                seizure_events.append({
                    "events_file": ef,
                    "row_index": idx,
                    "onset": row.get("onset"),
                    "duration": row.get("duration")
                })
        # check if sufficient seizures are recorded
        if len(seizure_events) >= min_n_sz:
            valid_subject_ids_.append(subject)



    #print(len(valid_subject_ids_), valid_subject_ids_)

    valid_patids[bids_root] = valid_subject_ids_

#print(valid_patids)

###### EXAMPLE CODE
##### load one file and attach the seizure info to the mne.epoch instance

 # subject = "01"   # zero-pad ie XX
 # session = "01"
 # run = "00"
 # epoch_length = 5.0  # seconds
 #
 # ## BIDS:: get the eeg and events file locations
 # edf_file = layout.get(
     # subject=subject,
     # session=session,
     # run=run,
     # suffix="eeg",
     # extension="edf",
     # return_type="file"
 # )[0]
 #
 # ## load in the MNE raw eeg signal instance
 # raw = mne.io.read_raw_edf(edf_file, preload=True)
 # # and apply some filtering
 # raw.filter(l_freq=1., h_freq=40.) # (optional)
 #
 # ### BIDS:: get the events file
 # events_file = layout.get(
     # subject=subject,
     # session=session,
     # run=run,
     # suffix="events",
     # extension="tsv",
     # return_type="file"
 # )[0]
 # # load the events file int
 # events_df = pd.read_csv(events_file, sep="\t")
 #
 # ## make mne annotation dict
 # annotations = mne.Annotations(
     # onset= events_df["onset"].values,
     # duration=events_df.get("duration", np.zeros(len(events_df))).values,
     # description=events_df["eventType"].astype(str).values,
 # )
 # # add to raw instance
 # raw.set_annotations(annotations)
 # ## Epoch data
 # epochs = mne.make_fixed_length_epochs(
     # raw,
     # duration=epoch_length,
     # overlap=0,
     # preload=True
 # )
 # labels = mne.events_from_annotations(raw)


def get_seizure_intervals(events_df):
    seizure_rows = events_df[
        events_df["eventType"].str.lower().str.contains("sz")
    ]

    intervals = []
    for _, row in seizure_rows.iterrows():
        start = row["onset"]
        end = row["onset"] + row.get("duration", 0)
        intervals.append((start, end))

    return intervals


##### URGENT CHECK THIS FUNCTION WRITE PROPER TEST!
#####
def label_epochs(epochs, seizure_intervals):
    #sfreq = raw.info["sfreq"] # BUG TODO this has to come from somewhere else!
    epoch_duration = epochs.tmax - epochs.tmin
    n_epochs = len(epochs)
    sfreq = 1 / epoch_duration

    labels = np.zeros(n_epochs, dtype=int)

    for i in range(n_epochs):
        t_start = epochs.events[i, 0] / sfreq
        t_end = t_start + epoch_duration

        # If any seizure interval overlaps this epoch → label = 1
        for sz_start, sz_end in seizure_intervals:
            if (t_start < sz_end) and (t_end > sz_start):
                labels[i] = 1
                break
    return labels


 #seizure_intervals = get_seizure_intervals(events_df)
 #labels = label_epochs(epochs, seizure_intervals)
 #np.sum(labels)
 #print("Interictal epochs:", np.sum(labels == 0))
 #print("Ictal epochs:", np.sum(labels == 1))
 #epochs.metadata = pd.DataFrame({"label": labels})
 #epochs["label == 1"]   # ictal
 #epochs["label == 0"]   # interictal
 #
 #