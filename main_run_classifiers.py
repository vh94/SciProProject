from src.events_to_annot import valid_patids
from run_single_subject import run_logreg_subjects
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# IO setup here
# .. define paths to BIDS DBS containing eeg data in (mne-readable) edf format
siena = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena'
chbmit = '/Volumes/Extreme SSD/EEG_Databases/BIDS_CHB-MIT'



def main(DB):
    subjects = valid_patids[DB]

    results = []

    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = {
            executor.submit(run_logreg_subjects, subject, DB): subject
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
    results = main(chbmit)
    print(results)
    results = pd.DataFrame(results)
    results.to_csv("all_subjects_results.csv")
