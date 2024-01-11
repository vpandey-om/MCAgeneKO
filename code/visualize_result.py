# Import necessary libraries for data processing, modeling, and evaluation
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 


result=pickle.load(open('/Users/vikash/gitlab/genekoai/deepModels/scripts/result_ensemble.p','rb'))
paras=[]
total_acc=[]
blood_acc=[]
liver=[]
male=[]
female=[]
sporo=[]
oocyst=[]
for k,v in result.items():
    total_acc.append(v[0][0][0])
    blood_acc.append(v[0][1][0])
    liver.append(v[0][1][1])
    male.append(v[0][1][2])
    female.append(v[0][1][3])
    sporo.append(v[0][1][4])
    oocyst.append(v[0][1][5])
    paras.append(k)
df=pd.DataFrame()
df['paras']=paras
df['total']=total_acc
df['blood']=blood_acc
df['male']=male
df['female']=female
df['sporozoite']=sporo
df['oocyst']=oocyst
df['liver']=liver

# Find index and maximum value for each column
for column in df.columns[1:]:
    max_value = df[column].max()
    index_of_max_value = df.loc[df[column].idxmax(),'paras']
    print(f"Column '{column}': Max value is {max_value} at index {index_of_max_value}")
column_names = ['total', 'blood', 'male', 'female', 'sporozoite', 'oocyst', 'liver']
data=df[column_names].values
# Sample row and column names
row_names = df['paras'].to_list()  # Add more row names as needed

# Find indices of maximum values along each column
max_indices = np.argmax(data, axis=0)

plt.figure(figsize=(20, 15))  # Set figure size
plt.imshow(data, cmap='viridis', interpolation='nearest')  # Display data matrix with viridis colormap

# Mask maximum values with a different color (e.g., red)

# Annotate each cell with its value and color the text based on the intensity of the value
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        value = data[i, j]
        # Set text color based on value intensity (brighter for higher values)
        text_color = 'white' if value < 0.5 else 'black'
        plt.text(j, i, f'{value:.2f}', color=text_color,ha='center', va='center', fontsize=10)


plt.colorbar()

for idx in range(data.shape[1]):
    plt.scatter(idx, max_indices[idx], color='red', marker='*', s=150,alpha=0.3)  # Star marker
    # plt.text(idx, max_indices[idx], f'{value:.2f}', color='red', ha='center', va='center', fontsize=15)

# Add color bar to show the scale


# Set title for the plot
plt.title('Parameter performance matrix',fontsize=18)

# Label for X-axis with column names
plt.xticks(np.arange(len(column_names)), column_names, rotation=90,fontsize=18)

# Label for Y-axis with row names
plt.yticks(np.arange(len(row_names)), row_names,fontsize=18)

plt.xlabel('Stages',fontsize=20)  # Label for X-axis
plt.ylabel('Parameters',fontsize=20)  # Label for Y-axis

# plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping

# plt.show()  # Show the plot
plt.savefig('data_mtraix.pdf', format='pdf')

