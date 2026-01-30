## Bids Derivatives for seizure prediction problems

In this project  patient specific logistic regression classifiers are used to either detect or
predict epileptic seizures with different occurrence periods in advance. 
Patient scalp EEG from to commonly used BIDS available databases, Siena and CHB_MIT.


### Patient selection

In this work only subjects (patients) with six or more annotated seizures are considered
This is done to ensure a sufficient number of three training seizures and at least three test seizures.


### Directory overview

```
├── README.md
├── analize_performance.py
├── environment.yml
├── idpinimg.png
├── main_create_features_labels.py
├── main_run_classifiers.py
├── notebooks
│   └── single_patient_example.ipynb
├── results
│   ├── all_subjects_results_pred.csv
│   ├── all_subjects_results_pred_3sz_train.png
│   └── report.md
├── src
│   ├── create_features.py
│   ├── events_to_annot.py
│   ├── linear_features.py
│   └── run_models.py
└── vis
    └── epoch_plots.py

```


#### Feature storage

All raw features are saved back into the BIDS Database in a BIDS derivate called  _linear_features_ 
as .npy numpy arrays for convieniece

### Data labelling

To distinguish between the two tasks 
1 detection , 2 prediction , another derivative seizures_pred is created 
which contains multiple .npy files of true labels for different types of problems detection windows, SPHs, SOPs can be
described

### Main files

These files are the ones you execute: 


There are two main files

- `main_create_features_labels.py`

which selects patients for study based on constraints (number of seizures), calculates features and labels for different tasks
and saves them as derivatives, and

- `main_run_classifiers.py`

which creates pseudoprospective train-test splits, trains and tests (evaluates) a set of classifiers for a given task.

Finally,

- `analaize_performace.py`

is a file that makes a boxplot showcasing the performance of the patient specific models for different tasks.

### Function files

- In `\src` the file `linear_features.py` holds the functions to calculate the 59 univariate linear features, it is used in 
to extract 59 univariate linear EEG features per channel.
They are a mix of time, and frequency domain features used in
previous EEG studies explained in https://doi.org/10.1002/epi4.12748

- The file `events_to_annot.py` implements functions to find the seizures, and make labels for prediction and detection.

- In `create_features.py` the methods from the previous two files to create the features and labels and store them
in the BIDS derivative directory.

- Finally `run_model.py` contains the method to load the features and labels form the Derivatives, make the train test split
and run the classifiers.


### Some results

Results are stored in the results dir. 
Atm they are a csv containing the prediction and detection metics for each patient, as well as a figure created from the `analyze_performance.py`
script.

The figure is:


![PR_AUC](results/all_subjects_results_pred_3sz_train.png)


### Issues and Discussion further ideas

Are in the `./results/report.md` file.


### Acknowledgements

The machine learning code and many of the principles of this project where inspired by:  
>Costa, G., Teixeira, C. & Pinto, M.F. Comparison between epileptic seizure prediction and forecasting based on machine learning. Sci Rep 14, 5653 (2024). https://doi.org/10.1038/s41598-024-56019-z


> Pinto MF, Batista J, Leal A, Lopes F, Oliveira A, Dourado A, et al. The goal of explaining black boxes in EEG seizure prediction is not to explain models' decisions. Epilepsia Open. 2023; 8: 285–297. https://doi.org/10.1002/epi4.12748 

![img.png](idpinimg.png)