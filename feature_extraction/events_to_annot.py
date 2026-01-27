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

def label_epochs_new(epochs, annotations):
    """
    Label epochs as seizure (1) or non-seizure (0) based on annotations.
    """
    # initialize labels
    y = np.zeros(len(epochs), dtype=np.int8)

    # epoch start/end in *samples*
    ep_start_samp = epochs.events[:, 0]
    ep_end_samp = ep_start_samp + epochs.time_as_index(
        epochs.tmax, use_rounding=False
    )[0]

    # annotation start/end in *samples*
    ann_start_samp = epochs.time_as_index(
        annotations.onset, use_rounding=False
    )
    ann_end_samp = epochs.time_as_index(
        annotations.onset + annotations.duration, use_rounding=False
    )

    # check overlap (vectorized)
    for a_start, a_end in zip(ann_start_samp, ann_end_samp):
        overlap = (ep_start_samp < a_end) & (ep_end_samp > a_start)
        y[overlap] = 1

    return y

def labels_for_detection(epochs, annotations):
    ep_start = epochs.events[:, 0][:, None]
    ep_end = ep_start + epochs.time_as_index(epochs.tmax)[0]

    ann_start = epochs.time_as_index(annotations.onset)[None, :]
    ann_end = epochs.time_as_index(
        annotations.onset + annotations.duration
    )[None, :]

    overlap = (ep_start < ann_end) & (ep_end > ann_start)
    return overlap.any(axis=1).astype(np.int8)

def labels_for_prediction(
    epochs,
    annotations,
    SOP=30 * 60,
    SPH=10 * 60,
):
    ep_start = epochs.events[:, 0]
    ep_end = ep_start + epochs.time_as_index(epochs.tmax)[0]

    y = np.zeros(len(epochs), dtype=np.int8)
    valid = np.ones(len(epochs), dtype=bool)

    SPH_samp = epochs.time_as_index(SPH)[0]
    SOP_samp = epochs.time_as_index(SOP)[0]

    seizure_onsets = epochs.time_as_index(annotations.onset)

    for onset in seizure_onsets:
        sop_start = onset - SOP_samp - SPH_samp
        sop_end = onset - SPH_samp

        # SOP labels
        y[(ep_start < sop_end) & (ep_end > sop_start)] = 1

        # invalidate SPH + ictal
        valid[ep_start >= sop_end] = False

    return y, valid

