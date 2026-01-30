def create_features_labels_single_subject(subject, DB) -> None :
    import numpy as np
    import pandas as pd
    import mne
    from bids import BIDSLayout
    from pathlib import Path
    from src.linear_features import univariate_linear_features, linear_feature_names
    from src.events_to_annot import (
        labels_for_detection, labels_for_prediction
    )
    """
    Contains Multithreading ready methods to handle a single subject from a BIDS database
    create_features_labels_single_subject :
    calculates 59 univariate linear Features using the univariate_linear_features defined in
    src/linear_features
    creates Labels for prediction at different SOPs for detection.
    Both features and labels for the different SOPs are saved as .npy or -npz files in the derivatives directory of the 
    BIDS database
    """

    print(f"--------------------- Subject {subject} ----------------------")

    # Recreate layout INSIDE process
    layout = BIDSLayout(DB, validate=False)
    all_edf_files = layout.get(subject=subject, extension='edf', return_type="file")
    all_tsv_files = layout.get(subject=subject, extension='tsv', return_type="file")

    # Make an identical dir to store the derivatives:::
    deriv_root = Path(DB) / "derivatives" / "linear_features"
    sub_path = deriv_root / f"sub-{subject}" / "eeg"
    sub_path.mkdir(parents=True, exist_ok=True)

    for edf_file, tsv_file in zip(all_edf_files, all_tsv_files):

        edf_file = Path(edf_file)
        edf = (
            mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            .filter(1., 40., verbose=False) # Dont filter each file ?!
        )
        events = pd.read_csv(tsv_file, sep="\t")
        print(f"{edf_file} has {len(events)} events")
        print(events)
        #annotations = mne.Annotations(
        #onset=events["onset"].values,
        #duration=events.get("duration", np.zeros(len(events))).values,
        #description=events["eventType"].astype(str).values,
        #)
        #edf.set_annotations(annotations)
        epochs = mne.make_fixed_length_epochs(
            edf, duration=5.0, preload=True, verbose=False
        )
        print("Calculating 59 univariate linear Features..")
        features_df = univariate_linear_features(epochs)
        # convert back to numpy, npz smaller than csv!
        X = features_df.to_numpy(dtype=np.float32)
        cols = features_df.columns.to_numpy() # cols are ch_name[feature names]  shape n:channels x nfeatures
        # ch_names = layout.get_channel_names(subject=subject, datatype="eeg")
        # Wirte the features to BIDS database derivatives linear_features
        feat_path = sub_path / f"{edf_file.stem}_linear-features.npz"
        np.savez(
            feat_path,
            X=X,
            cols=cols
        )
        ########  LABELLING PART
        print("Labeling epochs")
        labels_detection = labels_for_detection(epochs,events)
        # detection labels (.npy)
        det_path = sub_path / f"{edf_file.stem}_detection-labels.npy"
        np.save(det_path, labels_detection.astype(np.int8))
        # labels for prediciton SOP values in minutes see costa et al...
        sop_minutes = range(20, 55, 5)  # 20, 25, ..., 50
        SPH = 10 * 60  # seconds
        for sop_min in sop_minutes:
            SOP = sop_min * 60  # seconds

            print(f"Labeling prediction epochs | SOP={sop_min}min SPH=10min")

            labels_prediction, valid_tp = labels_for_prediction(
                epochs,
                events,
                SOP=SOP,
                SPH=SPH,
            )

            pred_path = sub_path / (
                f"{edf_file.stem}_prediction-labels_"
                f"SOP-{sop_min:02d}min_SPH-10min.npz"
            )

            np.savez(
                pred_path,
                y=labels_prediction.astype(np.int8),
                valid=valid_tp.astype(bool),
                SOP=SOP,
                SPH=SPH,
            )
    return None
