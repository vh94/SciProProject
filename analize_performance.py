import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
results_all = pd.read_csv("results/all_subjects_results_pred.csv")

#print(results_all)
# TODO:: A B C D FIGURE ROC CURVES
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

y_pred = results_all.predictions.values[1]
sns.boxplot(x="SOP", y="PR-AUC", data=results_all, fill = False, color = "black", ax = axes[0,0])
sns.stripplot(x="SOP", y="PR-AUC", data=results_all,hue="DB",size=4, dodge=False, ax = axes[0,0])

det_vs_pred = (results_all["SOP"] == 40) | (results_all["SOP"].isna())
sns.boxplot(x = "mode", y = "PR-AUC",data= results_all[det_vs_pred], fill = False, color = "black",ax = axes[0,1],legend = False)
sns.stripplot(x="mode", y="PR-AUC", data=results_all[det_vs_pred],hue="DB",size=4, dodge=False,alpha=0.6,ax = axes[0,1],legend = False)

sns.boxplot(x="SOP", y="AUC", data=results_all, fill = False, color = "black", ax = axes[1,0],legend = False)
sns.stripplot(x="SOP", y="AUC", data=results_all,hue="DB",size=4, dodge=False, ax = axes[1,0],legend = False)

det_vs_pred = (results_all["SOP"] == 40) | (results_all["SOP"].isna())
sns.boxplot(x = "mode", y = "AUC",data= results_all[det_vs_pred], fill = False, color = "black",ax = axes[1,1], legend = False)
sns.stripplot(x="mode", y="AUC", data=results_all[det_vs_pred],hue="DB",size=4, dodge=False,alpha=0.6,ax = axes[1,1],legend = False)
# grab handles + labels from one axis
handles, labels = axes[0,0].get_legend_handles_labels()
print(handles, labels)
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=1,
    bbox_to_anchor=(0.5, 0.0),
    title="Dataset"
)
axes[0,0].set(title="PR-AUC as a function of SOP")
axes[0,1].set(title="Detection vs prediction")
axes[0,0].legend_.remove()

#fig.tight_layout()
plt.tight_layout(rect=[0, 0.08, 1, 1])  # 👈 leave space at bottom
fig.savefig("./figures/all_subjects_results_pred.png")
