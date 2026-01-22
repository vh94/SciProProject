from bids import BIDSLayout
import pandas as pd
import mne
import pandas as pd
import numpy as np
import os

# The minuimum number of seizures to include the subject should be 4 ( 3 train and at least one test)
# I will default for a larger number tho in this project to reduce the number of individuals for comput. efficancy

# Parse all events for a subject to check if more than minseizures  seizures (3 trainig + 1 test) are avalaible and if
# the time differences are suffienct to use for the study
def get_valid_subject_ids(bids_root, min_n_sz = 4, eventType = "sz"):

    valid_subject_ids = []
    layout = BIDSLayout(bids_root)
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

            if "eventType" not in df.columns: # No events in this file
                continue
            # check if eventype matches:
            seizure_rows = df[df["eventType"].str.lower().str.contains(eventType)]
            # append
            for idx, row in seizure_rows.iterrows():
                seizure_events.append({
                    "events_file": ef,
                    "row_index": idx,
                    "onset": row.get("onset"),
                    "duration": row.get("duration")
                })
        # check if sufficient seizures are recorded:
        if len(seizure_events) >= min_n_sz:
            valid_subject_ids.append(subject)

    return valid_subject_ids

# This Dict stores the valid patient ids to be included:
def get_valid_subject_ids_multistudy(bids_roots,min_n_sz= 5):
    return { bids_root : get_valid_subject_ids(bids_root,min_n_sz ) for bids_root in bids_roots }

valid_patids = get_valid_subject_ids_multistudy(["/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena" ,"/Volumes/Extreme SSD/EEG_Databases/BIDS_CHB-MIT" ],min_n_sz = 4)
valid_patids



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
# THIS IS THE SINGLE MOST IMPORTANT FUNCTION
##### TODO

# We want to do two types of labels
# One for detection: ie class 1 = from onset till offset
# One for prediction: ie preictal phase is to be classified
# SOP: seizure onset period
# SPH: seizure prediction horizon
# Are epochs / annot fine like this??
def label_epochs_new(epochs,sfreq,annotations):
    # sfreq = edf.info["sfreq"]
    epoch_labels = np.zeros(len(epochs), dtype=np.int8)

    epoch_onsets = epochs.events[:, 0] / sfreq
    epoch_offsets = epoch_onsets + epochs.tmax - epochs.tmin

    for i, (ep_start, ep_end) in enumerate(zip(epoch_onsets, epoch_offsets)):
        for ann_start, ann_dur in zip(annotations.onset, annotations.duration):
            ann_end = ann_start + ann_dur

            # any overlap → seizure
            if ep_start < ann_end and ep_end > ann_start:
                epoch_labels[i] = 1
                break

    return epoch_labels
