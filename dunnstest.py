# Data Processing
import pandas as pd
import numpy as np

!pip install scikit-posthocs
# Modelling
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\Varsha\\OneDrive\\Desktop\\Varsha\\project data\\AIDS_Classification_50000.csv")

# Run Dunn's test with Bonferroni correction
dunn_results = sp.posthoc_dunn(df, val_col='time_to_liver_failure', group_col='trt', p_adjust='bonferroni')

# Display result
print("Dunn's test pairwise p-values with Bonferroni correction:")
print(dunn_results)

# Optional: Heatmap of p-values
sns.heatmap(dunn_results, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("Dunn's Test Pairwise Comparisons")
plt.show()