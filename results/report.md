# Report for the Epilepsy Prediction and Detection Project
Effective siezure prediction and fair comparision between models and patients is a complex endevour.

## Goals

1. The primary goal of the project was to compare the influence of the duration of the seizure occurence period on the  
the prediction accuracy across patient specific models from two databases.

2. The other focus of the project was to build a more robust, reproducible and performant end to end ML pipeline, mainly by leveraging the 
BIDS data structure and the possibility to store database derivatives


## Preliminary Results




![img.png](./figures/all_subjects_results_pred.png)

## Missing Pieces/Bugs/potential issues

- Elongation of the seizure occurence period (**SOP**) resulted in unforeseen issues for many patient timeseries
  - this lead to the loss of subjects in the highter SOP, due to the deletion of interictal 
- SIENA pat id 10 and 14 showed NaNs in the feature matrix 
  - While percentage of NaNs was very low (< 0.01%) this still hints to a problem with the feature calculation function
  - At this time NAN values where just replaced by the value 0.5 
- The EEG Data was bandpass filtered from 1 to 40Hz at a _per file level_ before concatination this could introduce some form of bias
- No re-referencing to the average montage nor any other preprocessing (artefact removal) was performed
- BIDS derivatives needs at least a JSON sidecar to be compliant


## Discussion



## Next Steps

- Event based statistics 
- more classifiers to compare 
  - Ensamble: ExtraTreesClassifier; SVM ; CNN
  
- Feature importance analysis using SHAP (Shapley Additive Explanations) 
  - for patient specific models
  - dissect, which type of feature form where (ie which 10-20 electrode) predicts preictal phases the best

