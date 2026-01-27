from feature_extraction.events_to_annot import get_seizure_intervals, label_epochs, valid_patids
from run_single_subject import run_single_subject
from concurrent.futures import ProcessPoolExecutor, as_completed

epoch_length = 5.0
sph = 10
window_length = 5

# IO setup here
# .. define paths to BIDS DBS containing eeg data in (mne-readable) edf format
siena = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena'
chbmit = '/Volumes/Extreme SSD/EEG_Databases/BIDS_CHB-MIT'



def main():
    DB = siena  # or chbmit
    subjects = valid_patids[DB]

    results = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_single_subject, subject, DB): subject
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
    results = main()
