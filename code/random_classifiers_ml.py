# Import necessary libraries for data processing, modeling, and evaluation
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, make_scorer,roc_auc_score
from sklearn.metrics import (accuracy_score, make_scorer,roc_auc_score, matthews_corrcoef, 
f1_score,precision_score, recall_score,confusion_matrix,roc_curve, auc)

import pickle
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import matplotlib.pyplot as plt

# Load data from a pickle file
data_output=pickle.load(open('data_output.p','rb'))

# Extract relevant data for modeling
y_true=data_output['reduced_Y'].values
y_true_copy=np.copy(y_true)
indices = np.where(y_true[:, 0] == 1)[0]





# Mask the data by replacing some values with -1

X=data_output['reduced_X']
for idx in indices:
    for j in range(1,y_true.shape[1]):
        if y_true[idx,j]==0:
            y_true[idx,j]=-1








# Initialize lists to store metrics and results
stage=[]
Percent=[]
Acc=[]
Roc=[]
Mcc=[]
F1=[]
Recall=[]
Precision=[]
confusion_mat=[]
models={}
fpr_list=[]
tpr_list=[]
auc_list=[]

# Define life stages for data and corresponding colors for plotting
life_stage=['blood','liver','male','female','sporozoite','oocyst']
color_list=['Blue',"#808080",'Green',"#D3D3D3","#F5F5F5","#2F4F4F"]

# Loop through each label in y_true to train and evaluate models
for label in range(y_true.shape[1]):
# for label in range(1,2):
    # Initialize Balanced Random Forest Classifier
    valid_indices = np.where(y_true[:, label] != -1)[0]
    tmp_y=y_true[valid_indices, label]
    tmp_x=X[valid_indices, :]
    stage.append(data_output['stages output'][label])
    Percent.append(tmp_y.sum()/len(tmp_y))

    # Split the data into training and test sets
    train_x, test_x, train_y, test_y = train_test_split( tmp_x,  tmp_y, test_size=0.2, random_state=42)

    # scoring = {'mcc': make_scorer(matthews_corrcoef)}

    # rf_pipeline = Pipeline([
    #     ('clf', BalancedRandomForestClassifier())
    # ])

    # Initialize and train the Balanced Random Forest Classifier
    brf = BalancedRandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42)
   
    brf.fit(train_x, train_y)

    # Predict on the test set and evaluate model performance
    y_preds = brf.predict(test_x)
    
    # Evaluate the accuracy of the model
    accuracy = accuracy_score(test_y, y_preds)
    mcc = matthews_corrcoef(test_y, y_preds)
    f1 = f1_score(test_y, y_preds)
    precision = precision_score(test_y, y_preds)
    recall = recall_score(test_y, y_preds)
    roc = roc_auc_score(test_y, y_preds)
    cm = confusion_matrix(test_y, y_preds)

    # Store metrics in respective lists

    Acc.append(accuracy)
    Mcc.append(mcc)
    F1.append(f1)
    Recall.append(recall)
    Precision.append(precision)
    Roc.append(roc)
    confusion_mat.append(cm)

  
    print(f"Balanced Random Forest Accuracy: {accuracy:.2f}")
    print(f"Balanced Random Forest ROC: {roc:.2f}")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(test_y, y_preds)

    # Calculate AUC score
    roc_auc = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(roc_auc)

# Create a DataFrame to display results
df=pd.DataFrame()
df['Stages']=life_stage
df['Rare class %']=Percent
df['Accuracy']=Acc
df['Correlation/Mcc']=Mcc
df['F1']=F1
df['Recall']=Recall
df['Precision']=Precision

# Print the results DataFrame
print(df.round(2))

print("DataFrame saved as an image successfully!")

# Plot ROC curves for each life stage

fsize=16
# Plot ROC curve
plt.figure(figsize=(8, 6))
for i in range(len(auc_list)):

    # plt.plot(fpr_list[i], tpr_list[i], color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot(fpr_list[i], tpr_list[i], color=color_list[i],label=f'ROC curve {life_stage[i]} (area = {auc_list[i]:.2f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
   
    plt.legend(loc='lower right')
    plt.grid(True)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# Set xlabel with increased font size
plt.xlabel('False Positive Rate', fontsize=fsize)  # Set fontsize to 14
plt.ylabel('True Positive Rate', fontsize=fsize)
# Adjust x-axis and y-axis ticks with increased font size
plt.xticks(fontsize=fsize)  # Set fontsize to 12 for x-axis ticks
plt.yticks(fontsize=fsize)  # Set fontsize to 12 for y-axis ticks
plt.title('Receiver Operating Characteristic (ROC) Curve',fontsize=fsize)
# Display the plot

plt.grid(True)
plt.tight_layout()  # Adjust layout to ensure labels and titles fit without overlapping
plt.show()




