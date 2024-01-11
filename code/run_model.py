# Import necessary libraries
from model import MultiInputNN2 
from sklearn.model_selection import train_test_split, StratifiedKFold,GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle 
import numpy as np 
# from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import BaggingClassifier
import tensorflow as tf
from tensorflow.keras.callbacks import Callback



# Load training and testing data from picke file
data_output=pickle.load(open('data_output.p','rb'))



# Function to create a Keras model using MultiInputNN2

def create_model(cnode=10, pnode=100, gnode=10, gonode=100):
    model = MultiInputNN2(data_output, cnode=cnode, pnode=pnode, gnode=gnode, gonode=gonode).model
    return model


# Extract classes for multilevel output

y_true=data_output['reduced_Y'].values
y_true_copy=np.copy(y_true)
indices = np.where(y_true[:, 0] == 1)[0]


# Mask some values in y_true for specific conditions (data is not avilable for other stages than blood )
for idx in indices:
    for j in range(1,y_true.shape[1]):
        if y_true[idx,j]==0:
            y_true[idx,j]=-1



# Define a grid of parameters for hyperparameter tuning
param_grid_NN = {
    'cnode': [1,2,3],
    'pnode':[2,4,8,16],
    'gnode': [2,4,8],
    'gonode': [8,10,16,32],
    'drop_rate':[0,0.05,0.1,0.2,0.3]
}




# Set the number of estimators for Bagging Classifier
n_estimators=30
result={}

# Iterate through all combinations of hyperparameters

for cnode in param_grid_NN['cnode']:
    for pnode in param_grid_NN['pnode']:
        for gnode in param_grid_NN['gnode']:
            for gonode in param_grid_NN['gonode']:
                for dpr in param_grid_NN['drop_rate']:

                    # Initialize lists to store estimators and predictions
                    esimators=[]
                    predictions=[]
                    # Create Bagging Classifier with multiple estimators
                    for est_i in range(n_estimators):
                        y_true_mask=np.copy(y_true)
                        
                        
                        # Create bootstrap samples for each stage
                        # make sure classes 0 and 1 are balanced 
                        for stage_i in range(y_true_mask.shape[1]):
                            # indices_1=np.where(y_true_mask[y_true_mask[:,stage_i]!=-1,stage_i]==1)
                            # indices_0=np.where(y_true_mask[y_true_mask[:,stage_i]!=-1,stage_i]==0)
                            indices_1=np.where(y_true_mask[:,stage_i]==1)
                            indices_0=np.where(y_true_mask[:,stage_i]==0)
                            print('original',len(indices_1[0]),len(indices_0[0]),y_true_mask.shape[0],stage_i)
                            ## get smaple of diffrence between indices 
                            diff_num=len(indices_0[0])-len(indices_1[0])
                            mask_indices=np.random.choice(indices_0[0], size= diff_num, replace=False)
                            y_true_mask[mask_indices,stage_i]=-1
                            indices_1=np.where(y_true_mask[:,stage_i]==1)
                            indices_0=np.where(y_true_mask[:,stage_i]==0)
                            print('masked',len(indices_1[0]),len(indices_0[0]),y_true_mask.shape[0],stage_i)
                        
                        
                        # Split the data into train, validation, and test sets
                        # Splitting the data into 60% training and 40% temporary (validation + test)
                        
                        train_x, temp_x, train_y, temp_y = train_test_split(data_output['reduced_X'], y_true_mask, test_size=0.4, random_state=42)

                        # Splitting the temporary data (which is 40% of the original data) into 50% validation and 50% test
                        val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)
                        # Initialize and fit the model
                        mod1 = MultiInputNN2(data_output,cnode=cnode, pnode=pnode, gnode=gnode, gonode=gonode,drop_rate=dpr)
                        ### change x input in the list
                        train_input_data={}
                        val_input_data={}
                        test_input_data={}

                        for k,v in mod1.input_original_index.items():
                            train_input_data[k]=train_x[:,v]
                            val_input_data[k]=val_x[:,v]
                            test_input_data[k]=test_x[:,v]
                            has_nan = np.isnan(train_input_data[k]).any()
                            
                        
                        mod1.fit(train_input_data, train_y, epochs=50, batch_size=32, validation_data=(val_input_data,val_y))
                        # save all estimators in the list
                        esimators.append(mod1)
                        ## save the model
                        # Predict using the current estimator
                        prediction=esimators[est_i].cal_predict_prob(test_input_data)
                        
                        predictions.append(prediction)
                    # Compute ensemble predictions and evaluate accuracy
                    # Apply soft-voting by averaging predicted probabilities across models
                    ensemble_predictions_proba = np.mean(predictions, axis=0)

                    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
                    # threshold = 0.5
                    # ensemble_predictions = (ensemble_predictions_proba > threshold).astype(int)

                    accuracy_test=mod1.cal_stage_accuracy(ensemble_predictions_proba,test_y)
                    
                      
                    # Save the model if the accuracy meets the criteria
                    if accuracy_test[1][0]>0.67:

                        filename_elements = [str(num).replace('.', '_') for num in [cnode,pnode,gnode,gonode,dpr]]
                        filename = 'file_ensemble2' + '_'.join(filename_elements) + '.pkl'


                        with open(filename, 'wb') as f:
                            pickle.dump(esimators, f)

                 
                    # Store the results in a dictionary for all hyperparameters
                    result[(cnode,pnode,gnode,gonode,dpr)]=[accuracy_test]
                    with open('result_ensemble2.p', 'wb') as f:
                        pickle.dump(result, f)

                    # # with open(filename, 'rb') as f:
                    # #     loaded_model = pickle.load(f)
                    # print(accuracy_test[1][0])



