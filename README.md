## Bids Derivatives for seizure prediction problems

### Data

This project is intended for use with eeg data in BIDS format, it 
was developed using two scalp EEG databases Siena and CHB_MIT 

### Patient selection

In this work only subjects (patients) with more than five annotated seizures are considered
This is done to ensure a sufficient number of three training seizures and at least two test seizures. 
### Feature calculation

The project contains a function (see: `./feature_extraction/linear_features.py`)
to extract 59 univariate linear EEG features per channel. They are a mix of time, and frequency domain features used in 
previous EEG studies


#### Feature storage

All raw features are saved back into the BIDS Database in a BIDS derivate called  _linear_features_ 
as .npy numpy arrays for convieniece

### Data labelling

To distinguish between the two tasks 
1 detection , 2 prediction , another derivative seizures_pred is created 
which contains multiple .npy files of true labels for different types of problems detection windows, SPHs, SOPs can be
described





The prediction shift controls the problem type it is used to label the True positive 
timepoints for the detection problem they are all ictal timepoints, for the predictiton they fall into the seizure
```
for database do:
    for subject do:
        # calculate feature Matrix
        X <- (59 univarate features x N channels) x (time x f)/ window_length  
        
        for prediction_shift in tasks do:
            # label TP windows:
            y <- t_ictal + prediction_shift
            
            # 3 Pseudoprospective tt - split :
            train <- (y,X)[until end postictal seizure (type) 3]
            test <- (y,X)[all subsequent]
            
            # train patient specific AutoML model:
            clf <- tpot_GA
            for gen in generations do:
                clf <<- evolve(clf, clf.train(train))
                end
            # Save best model stats for task
            metrics.append(clf.score(test), clf.topFeatures)
            # store train test data 
            train_all.append(train)
            test_all.append(test) 
        end
    end       
```
or train a global model
hard TODO since, number and channel names, sampling freq, etc might differ !
```
train_all_tasks = train_all.reshape(len(all_tasks), )?
clf <- tpot_GA
clf <<- evolve(clf, clf.train(train_all))    
metrics.append(clf.score(test_all)) 
```



#### Main files
These files are the ones you execute:

#### Function files



The machine learning code and may of the pricples of this project where inspired by  
Costa, G., Teixeira, C. & Pinto, M.F. Comparison between epileptic seizure prediction and forecasting based on machine learning. Sci Rep 14, 5653 (2024). https://doi.org/10.1038/s41598-024-56019-z

## Please cite this work as:
