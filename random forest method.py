# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("C:\\Users\\Varsha\\OneDrive\\Desktop\\Varsha\\project data\\AIDS_Classification_50000.csv")

#label high/low risk
df['cd4_drop'] = df['cd40'] - df['cd420']
df['risk_label'] = df['cd4_drop'].apply(lambda x: 1 if x >= 100 else 0)
#to label patients as: High-risk: Rapid progression (e.g., large CD4 drop)
#Low-risk: Stable/slower progression
#1 = High-risk (large CD4 drop)
#0 = Low-risk

#Define features and target
features = ['cd40', 'cd420', 'cd80', 'cd820', 'drugs', 'homo', 'hemo', 'age']
X = df[features]
y = df['risk_label']

#standardising
X_scaled = StandardScaler().fit_transform(X)

# 6. Train/Test split
#The training data is used to fit the model. 
# The algorithm uses the training data to learn the relationship between the features and the target. 
# The test data is used to evaluate the performance of the model.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
#This line splits the dataset into training and testing sets 
#a critical step in machine learning to evaluate model performance on unseen data.
#train_test_split(...):A function from sklearn.model_selection that randomly splits 
# the data into training and test sets.
#y:target variable, e.g., risk_group (0 = low-risk, 1 = high-risk).
#test_size=0.3: Means 30% of the data will be used for testing, and 70% for training.
# random_state=42:ensure the split is reproducible. You’ll get the same results each time you run it.
#X_train, X_test: Features for training and testing.
#y_train, y_test:Target values for training and testing.

# 7. Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
#n_estimators=100:to build 100 decision trees in the forest (more trees = more robust, to a point).
rf.fit(X_train, y_train)
#The model:
#Builds 100 decision trees on different samples of your data.
#At each split, it chooses a subset of features randomly to decide the best split.
#It aggregates the results from all the trees.
#Overall Accuracy→99.7% (14,934/15,000 correct)

#At this point, we have a trained random forest model, 
# but we need to find out whether it makes accurate predictions.
# 8. Predict
y_pred = rf.predict(X_test)#make predictions on test data.

#The simplest way to evaluate this model is using accuracy;
# we check the predictions against the actual values in the test set and count up how many the model got right.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 9. Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
#result interpretation
#[[13760 14]→Class 0 (Low-Risk): 13,760 correct, 14 misclassified
#[ 52  1174]]→Class 1 (High-Risk): 1,174 correct, 52 misclassified
#Precision for high-risk (0.99)→99% of those predicted as high-risk truly are
#Recall for high-risk (0.96)→96% of actual high-risk patients were correctly identified.
#Overall Accuracy:99.7% (14,934/15,000 correct)