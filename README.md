## Comparison-Between-Epileptic-Seizure-Prediction-and-Forecasting

>We would like to test if the seizure prediction problem from EEG has a general solution in
form of a overall best model or is instead rather a patients specfic one ie ideal features and best type of classifier
vary from subject to subject.


>We would like to know the influence of dataset selection, epilepsy type and problem type,
> ie prediction vs detection
> on the performance of classifiers and features.

,
and many other factors such as
dataset, specfic epilepsy and seizure type profile 

### overview

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
#

clf <- tpot_GA
clf <<- evolve(clf, clf.train(train_all))    
metrics.append(clf.score(test_all)) 
```



#### Main files
These files are the ones you execute:

#### Function files



## Please cite this work as:

Costa, G., Teixeira, C. & Pinto, M.F. Comparison between epileptic seizure prediction and forecasting based on machine learning. Sci Rep 14, 5653 (2024). https://doi.org/10.1038/s41598-024-56019-z
