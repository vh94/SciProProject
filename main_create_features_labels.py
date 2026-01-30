from src.create_features import create_features_labels_single_subject
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.events_to_annot import get_valid_subject_ids_multistudy
# IO setup here
# .. define paths to BIDS DBS containing eeg data in (mne-readable) edf format
siena = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena'
chbmit = '/Volumes/Extreme SSD/EEG_Databases/BIDS_CHB-MIT'

## Select patient with at leat 5 seizures
valid_patids = get_valid_subject_ids_multistudy([siena, chbmit], min_n_sz=5)


def main(DB, subjects):

    results = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(create_features_labels_single_subject, subject, DB): subject
            for subject in subjects
        }

        for future in as_completed(futures):
            subject = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Subject {subject} failed: {e}")

    print("All subjects complete")
    return results

if __name__ == "__main__":

    main(siena, valid_patids[siena]) # results = noting in this case all wirtes to BIDS deriatives ... ? shold write some retuncodes...
    main(chbmit,valid_patids[chbmit]) # results = noting in this case all wirtes to BIDS deriatives ... ? shold write some retuncodes...
