from src.events_to_annot import get_valid_subject_ids_multistudy
from src.run_models import run_logreg_subjects
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# IO setup here
# .. define paths to BIDS DBS containing eeg data in (mne-readable) edf format
siena = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena'
chbmit = '/Volumes/Extreme SSD/EEG_Databases/BIDS_CHB-MIT'
valid_patids = get_valid_subject_ids_multistudy([siena, chbmit], min_n_sz=5)


def main_detection(DB):
    subjects = valid_patids[DB]

    results = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(run_logreg_subjects, subject, DB, nseizures_train=3): subject
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

def main_prediction(DB, SOP):
    subjects = valid_patids[DB]

    results = []

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(run_logreg_subjects, subject, DB, mode = "prediction", SOP = SOP, nseizures_train=3): subject
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

    print("------------- RUNNING DETECTION TASKS -------------")
    print(f"Detecting subjects...{valid_patids}")
    print("------- CHB-MIT")
    results_chbmit = main_detection(chbmit)
    results_chbmit = pd.DataFrame([re for re in results_chbmit if re is not None])  # things might go wrong, i will look into it ; but for now we just go on
    results_chbmit["DB"] = "chbmit"
    print("------- DETECTION ------ SIENA")
    results_siena = main_detection(siena)
    results_siena = pd.DataFrame([re for re in results_siena if re is not None])
    results_siena["DB"] = "siena"

    results_all = pd.concat([results_chbmit, results_siena])

    print("------------- RUNNING PREDICTION TASKS -------------")

    for SOP in range(20, 55, 5):
        print(f"--------------------- RUNNING PREDICTION for SOP {SOP} -------------")
        results_chbmit = main_prediction(chbmit, SOP = SOP )
        results_chbmit = pd.DataFrame([x for x in results_chbmit if x is not None])
        print("---------- CHB-MIT")
        results_chbmit["DB"] = "chbmit"
        results_chbmit["SOP"] = SOP
        # Subject 10 failed: Input X contains NaN.
        # TODO:: FIx BUGS: either no seizure starts found SOP overlaps?!
        #  or the issue is:
        #  Input X contains NaN.


        results_siena = main_prediction(siena, SOP = SOP)
        results_siena = pd.DataFrame([x for x in results_siena if x is not None])
        print("---------- SIENA")
        results_siena["DB"] = "siena"
        results_siena["SOP"] = SOP
        results_all = pd.concat([results_all, results_chbmit, results_siena])

    results_all.to_csv("./results/all_subjects_results_pred.csv")

