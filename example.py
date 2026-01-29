from matplotlib import pyplot as plt

from run_single_subject import run_logreg_subjects
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, pair_confusion_matrix
siena = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena'
chbmit = '/Volumes/Extreme SSD/EEG_Databases/BIDS_CHB-MIT'
subject = "04"

results = pd.DataFrame([run_logreg_subjects( "14",siena,mode="prediction",SOP = 45)])
results
results = pd.DataFrame([run_logreg_subjects( s,chbmit,mode="prediction",SOP = 25)])

cfm = results.confusion_matrix.values[0]
disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
disp.plot()
plt.show()