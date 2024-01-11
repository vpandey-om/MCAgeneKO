
''''
The provided code appears to define a neural network architecture using 
TensorFlow and Keras for multi-input data. The architecture is structured to handle 
various types of input data such as cells, protein sequences, gene sequences, and gene ontology features.
'''
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda,concatenate,Dropout
# from datautil import data_output
import numpy as np 
from sklearn.metrics import roc_auc_score
# Custom Lambda function to extract columns based on indices


# Custom function to extract specific columns from a tensor based on indices
def extract_columns(x, indices):
    return tf.gather(x, indices, axis=1)


# # # Custom loss function with data masking
# # def masked_binary_crossentropy(y_true, y_pred):
    
# #     mask = tf.math.not_equal(y_true, -1)
# #     loss = tf.keras.losses.binary_crossentropy(tf.where(mask, y_true, tf.zeros_like(y_true)), y_pred)
# #     masked_loss = tf.where(mask, loss, tf.zeros_like(loss))
    
# #     return tf.reduce_sum(masked_loss) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32))

# Function to compute class weights for handling imbalanced datasets
def compute_class_weights(labels, epsilon=1e-7):
    """
    Compute class weights based on the frequency of outcomes for each label.
    
    Args:
    - labels: True labels, shape (num_samples, num_classes).
    - epsilon: Small value to avoid division by zero.
    
    Returns:
    - class_weights: Computed class weights, shape (num_classes,).
    """
    # Mask where labels are not equal to the mask_value
    mask = tf.math.not_equal(labels, -1)
    
    # Compute class frequencies excluding mask values
    class_frequencies = tf.reduce_sum(tf.cast(mask, tf.float32), axis=0) / tf.cast(tf.shape(labels)[0], tf.float32)
    
    # Add epsilon to avoid division by zero
    class_frequencies = class_frequencies + epsilon
    
    # Compute class weights as the inverse of class frequencies
    total_classes = tf.reduce_sum(tf.ones_like(class_frequencies))
    class_weights = total_classes / (class_frequencies * total_classes)
    
    # # Compute class frequencies (sum across all samples and then normalize)
    # class_frequencies = tf.reduce_sum(labels, axis=0) / tf.cast(tf.shape(labels)[0], tf.float32)
    
    # # Add epsilon to avoid division by zero
    # class_frequencies = class_frequencies + epsilon
    
    # # Compute class weights as the inverse of class frequencies
    # total_classes = tf.reduce_sum(tf.ones_like(class_frequencies))
    # class_weights = total_classes / (class_frequencies * total_classes)
    
    return class_weights





def masked_binary_crossentropy_weight(y_true, y_pred):
    '''Custom loss function: Binary cross-entropy with data masking and class weights'''
    # def loss(y_true, y_pred):
    # Mask where y_true is not -1
    # Define class weights (e.g., higher weight for class 1)
    # class_weights = tf.constant([1.0, weight])  # Adj
    class_weights = compute_class_weights(y_true)  # Class weights for each label
    mask = tf.math.not_equal(y_true, -1)
    # Cast the tensor from int64 to float32
    y_true = tf.cast(y_true, dtype=tf.float32)

    # Compute binary cross-entropy loss element-wise
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    bce_loss = - (y_true * tf.math.log(y_pred + 1e-15) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-15))
    
    # Apply mask to ignore -1 values
    masked_loss = tf.where(mask, bce_loss, tf.zeros_like(bce_loss))

    # Apply class weights to the loss
    
    # Ensure compatible dimensions for class_weights and masked_loss
    
    # weighted_loss = tf.math.multiply(masked_loss, class_weights)
    # Expand dimensions of class_weights to match masked_loss for element-wise multiplication
    # Expand dimensions of class_weights to match masked_loss for element-wise multiplication
    expanded_class_weights = tf.expand_dims(class_weights, axis=0)
    
    # Apply class weights to the masked loss
    weighted_loss = tf.math.multiply(masked_loss, expanded_class_weights)
    
    # Compute mean loss over valid entries (ignoring -1)
    loss = tf.reduce_sum(weighted_loss, axis=1) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=1)
    
    # loss1 = tf.reduce_sum(masked_loss, axis=1) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=1)
    # Ensure both loss and loss1 have the same shape


    return loss

def masked_binary_crossentropy(y_true, y_pred):
    '''Custom loss function: Binary cross-entropy with data masking '''
    # Mask where y_true is not -1
    # Define class weights (e.g., higher weight for class 1)
    
    
    mask = tf.math.not_equal(y_true, -1)
    # Cast the tensor from int64 to float32
    y_true = tf.cast(y_true, dtype=tf.float32)
    # has_nan = tf.math.reduce_any(tf.math.is_nan(y_true))
    # has_nan1 = tf.math.reduce_any(tf.math.is_nan(y_true))
    
    # Compute binary cross-entropy loss element-wise
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
   
    bce_loss = - (y_true * tf.math.log(y_pred ) + (1 - y_true) * tf.math.log(1 - y_pred ))
    
    # Apply mask to ignore -1 values
    masked_loss = tf.where(mask, bce_loss, tf.zeros_like(bce_loss))
    
    
    # Compute mean loss over valid entries (ignoring -1)
    loss = tf.reduce_sum(masked_loss, axis=1) / (tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=1)+epsilon)
    # print("loss",loss)
    
    return loss

# Register the custom loss function with Keras
tf.keras.utils.get_custom_objects().update({'masked_binary_crossentropy': masked_binary_crossentropy})

class MultiInputNN2:
    '''# Class definition for the MultiInput Neural Network'''
    ## cnode is the number of dense nodes in cells layer 
    ## pnode is the number of dense nodes in protein  layer 
    ## gnode is the number of dense nodes in gene layer 
    ## gonode is the number of dense nodes in gene ontology layer 
    def __init__(self, data_dict, cnode=10,pnode=100,gnode=10,gonode=100,drop_rate=0.5,minor_weight=2.0):

        '''
        
        Initialize the MultiInput Neural Network model.
        
        Args:
        - data_dict: Dictionary containing data for different features.
        - cnode, pnode, gnode, gonode: Number of nodes in various layers.
        - drop_rate: Dropout rate for regularization.
        - minor_weight: Weight for minor classes in the loss function.
        
        '''
        self.out_stages=['blood','liver','male','female','sporo','oocyst']
        self.cnode=cnode
        self.pnode=pnode
        self.gnode=gnode
        self.gonode=gonode
        self.drop_rate=drop_rate
        self.data_dict = data_dict
        self.minor_weight = minor_weight
        ### build single cells branch models
        self.build_single_cells_branch()
        ## build other branches such as protein,gene,go based features 
        # self.build_other_branch()
        self.model = self.build_model()


    def build_single_cells_branch(self):
        """
        Build the neural network branch for cells-based features.
        
        This method constructs the architecture for processing cells-based data, 
        including dropout layers and dense layers.
        """
        ###
        print('getting into cells neural networks')
        inputs=[]
        outputs=[]
        input_combined = Input(shape=(self.data_dict['reduced_X'].shape[1],), name='input_combined')
        input_original_index={}
        drop_rate=self.drop_rate

        for k,v in self.data_dict['stages cells index'].items():
            
            # Sanitize the name 'k' to make it a valid layer name
            sanitized_name = k.replace(' ', '_').replace('-', '_').replace('.', '_').lower()
            input_layer=Input(shape=(len(v),),name=sanitized_name)
            inputs.append(input_layer)
            ## build outputs
            # dense_layer = Dense(self.cnode, activation='relu')(input_layer) ### 5 can be a parameter
            # dropout_layer = Dropout(drop_rate)(dense_layer)  # Adding a dropout layer with a dropout rate of 0.5
            # outputs.append(dropout_layer)
            dropout_layer = Dropout(drop_rate)(input_layer)
            dense_layer = Dense(self.cnode, activation='relu')(dropout_layer) ### 5 can be a parameter
              # Adding a dropout layer with a dropout rate of 0.5
            outputs.append(dense_layer)
            input_original_index[sanitized_name]=v

        ## build for protein based features
        
        lowerIdx=len(self.data_dict['cells'])
        upperIdx=len(self.data_dict['cells'])+len(self.data_dict['protein_features'])
        Idx=list(range(lowerIdx,upperIdx ))
        input_layer=Input(shape=(len(self.data_dict['protein_features']),),name='proteinsequence')
        inputs.append(input_layer)
        #input_layer = Lambda(lambda x: x[:, Idx],name='protein sequence')(input_combined)
        # input_layer = Lambda(lambda x: extract_columns(x, Idx), 
        #                   arguments={'indices': Idx}, 
        #                   name='proteinsequence')(input_combined)
        # input_layer = Lambda(lambda x, indices=Idx: extract_columns(x, indices), 
        #                         name='proteinsequence')(input_combined)

        # dense_layer = Dense(self.pnode, activation='relu')(input_layer) ### 5 can be a parameter
        # # outputs.append(dense_layer)
        # dropout_layer = Dropout(drop_rate)(dense_layer)  # Adding a dropout layer with a dropout rate of 0.5
        # outputs.append(dropout_layer)

        dropout_layer = Dropout(drop_rate)(input_layer)
        dense_layer = Dense(self.pnode, activation='relu')(dropout_layer) ### 5 can be a parameter
        # outputs.append(dense_layer)
          # Adding a dropout layer with a dropout rate of 0.5
        outputs.append(dense_layer)
        input_original_index['proteinsequence']=Idx

        ## build for  gene sequence
        
        lowerIdx=upperIdx
        upperIdx=upperIdx+len(self.data_dict['gene_features'])
        Idx=list(range(lowerIdx,upperIdx ))

        # input_layer = Lambda(lambda x, indices=Idx: extract_columns(x, indices), 
        #                         name='genesequence')(input_combined)
        input_layer=Input(shape=(len(self.data_dict['gene_features']),),name='genesequence')
        inputs.append(input_layer)

        # dense_layer = Dense(self.pnode, activation='relu')(input_layer) ### 5 can be a parameter
        # #outputs.append(dense_layer)
        # dropout_layer = Dropout(drop_rate)(dense_layer)  # Adding a dropout layer with a dropout rate of 0.5
        # outputs.append(dropout_layer)


        dropout_layer = Dropout(drop_rate)(input_layer)
        dense_layer = Dense(self.pnode, activation='relu')(dropout_layer) ### 5 can be a parameter
        # outputs.append(dense_layer)
          # Adding a dropout layer with a dropout rate of 0.5
        outputs.append(dense_layer)

        input_original_index['genesequence']=Idx

        ## build for  go features
        
        lowerIdx=upperIdx
        upperIdx=upperIdx+len(self.data_dict['gene_ontology_features'])
        Idx=list(range(lowerIdx,upperIdx ))
        #input_layer = Lambda(lambda x: x[:, Idx],name='gene ontology')(input_combined)
        # input_layer = Lambda(lambda x, indices=Idx: extract_columns(x, indices), 
        #                         name='geneontology')(input_combined)

        input_layer=Input(shape=(len(self.data_dict['gene_ontology_features']),),name='geneontology')
        inputs.append(input_layer)

        # dense_layer = Dense(self.pnode, activation='relu')(input_layer) ### 5 can be a parameter
        # #outputs.append(dense_layer)
        # dropout_layer = Dropout(drop_rate)(dense_layer)  # Adding a dropout layer with a dropout rate of 0.5
        # outputs.append(dropout_layer)


        dropout_layer = Dropout(drop_rate)(input_layer)
        dense_layer = Dense(self.pnode, activation='relu')(dropout_layer) ### 5 can be a parameter
        # outputs.append(dense_layer)
          # Adding a dropout layer with a dropout rate of 0.5
        outputs.append(dense_layer)

        input_original_index['geneontology']=Idx


        self.inputs=inputs
        self.outputs=outputs
        self.input_original_index=input_original_index
        return inputs, outputs

    def build_other_branch(self):
        """
        Build other neural network branches for protein, gene, and gene ontology features.
        
        This method constructs the architecture for processing other types of features.
        """
         
        ## build for protein based features
        input_layer=Input(shape=(len(self.data_dict['protein_features']),),name='protein sequence')
        self.inputs.append(input_layer)
        ## build outputs
        dense_layer = Dense(self.pnode, activation='relu')(input_layer) ### 5 can be a parameter
        self.outputs.append(dense_layer)
        upperIdx=len(self.data_dict['cells'])+len(self.data_dict['protein_features'])
        #self.input_original_index.append(list(range(len(self.data_dict['cells']),upperIdx )))
        self.input_original_index['protein sequence']=list(range(len(self.data_dict['cells']),upperIdx ))
        
        ### build for gene based features 
        input_layer=Input(shape=(len(self.data_dict['gene_features']),),name='gene sequence')
        self.inputs.append(input_layer)
        ## build outputs
        dense_layer = Dense(self.gnode, activation='relu')(input_layer) ### 5 can be a parameter
        self.outputs.append(dense_layer)
        lowerIdx=upperIdx
        upperIdx=upperIdx+len(self.data_dict['gene_features'])
        #self.input_original_index.append(list(range(lowerIdx,upperIdx))) 
        self.input_original_index['gene sequence']=list(range(lowerIdx,upperIdx))

        ### build for gene ontology features 
        input_layer=Input(shape=(len(self.data_dict['gene_ontology_features']),),name='gene ontology')
        self.inputs.append(input_layer)
        ## build outputs
        dense_layer = Dense(self.gonode, activation='relu')(input_layer) ### 5 can be a parameter
        self.outputs.append(dense_layer)
        lowerIdx=upperIdx
        upperIdx=upperIdx+len(self.data_dict['gene_ontology_features'])
        # self.input_original_index.append(list(range(lowerIdx,upperIdx))) 
        self.input_original_index['gene ontology']=list(range(lowerIdx,upperIdx))

    def build_model(self):
        """
        Construct the complete MultiInput Neural Network model.
        
        This method combines all the branches and creates a unified model.
        """
    
        # Merge All Branches
        merged = concatenate(self.outputs)

         # # Additional Dense Layers
        # dense1 = Dense(16, activation='relu')(merged)

        # Output layer for multi-class classification
        #output = Dense(len(self.out_stages), activation='softmax')(merged)  # 3 classes
        output = Dense(len(self.out_stages), activation='sigmoid')(merged)  # 3 classes
        

        # # Additional Dense Layers
        # dense1 = Dense(32, activation='relu')(merged)

        # final_outputs=[]
        
        # for item in self.out_stages:
        #     output_layer = Dense(1, activation='softmax', name=item)(dense1)
        #     final_outputs.append(output_layer)

    

        # Create output for each stage 
        model = Model(inputs=self.inputs, outputs=output)
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='masked_binary_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer='adam', loss=masked_binary_crossentropy_weight, metrics=['accuracy'])
        return model

    def fit(self, train_x, train_y, epochs=10, batch_size=32, validation_split=0.2,validation_data=None):
        '''fit the model'''
        if validation_data:
            self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)  
        else:            
            self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        
    def evaluate(self, test_x, test_y):
        ''' evaluate the model'''
        # Predict on test data
        predictions = self.model.predict(test_x)
    
        # Apply the mask on the predictions
        masked_predictions = np.where(test_y == -1, -1, (predictions > 0.5).astype(np.int64))
        
        # Calculate accuracy
        accuracy = np.mean(np.equal(masked_predictions, test_y))

        # Evaluate the model using the evaluate method and return loss and accuracy
        return accuracy
    

    def predict(self, test_x, test_y):
        ''' predict the model'''
        # Predict on test data
        predictions = self.model.predict(test_x)
        
        # Apply the mask on the predictions
        masked_predictions = np.where(test_y == -1, -1, (predictions > 0.5).astype(np.int64))
        
        # Calculate accuracy
        accuracy = np.mean(np.equal(masked_predictions, test_y))

        # Evaluate the model using the evaluate method and return loss and accuracy
        return accuracy
    

    def cal_predict_prob(self, test_x):
        # Predict on test data
        ''' calculate the model prediction probability'''
        predictions = self.model.predict(test_x)

        return predictions

    def cal_accuracy(self, test_x, test_y):
        ''' calculate the model prediction accuracy'''
        # Predict on test data
        
        predictions = self.model.predict(test_x)
        
        # Apply the mask on the predictions
        masked_predictions = np.where(test_y == -1, -1, (predictions > 0.5).astype(np.int64))
        
        # Calculate accuracy
        # accuracy = np.mean(np.equal(masked_predictions, test_y),axis=0)
        # mean_accuracy = np.mean(np.equal(masked_predictions, test_y))
        # Initialize lists to store metrics for each label
        precisions = []
        recalls = []
        f1_scores = []
        auc_scores = []  # Initialize list to store AUC scores
        accuracies=[]
        
        # Calculate metrics for each label
        for i in range(test_y.shape[1]):  # Assuming test_y.shape[1] gives the number of labels/columns
            true_positive = np.sum((masked_predictions[:, i] == 1) & (test_y[:, i] == 1))
            false_positive = np.sum((masked_predictions[:, i] == 1) & (test_y[:, i] == 0))
            false_negative = np.sum((masked_predictions[:, i] == 0) & (test_y[:, i] == 1))
            true_negative = np.sum((masked_predictions[:, i] == 0) & (test_y[:, i] == 0))

            precision = true_positive / (true_positive + false_positive + 1e-9)
            recall = true_positive / (true_positive + false_negative + 1e-9)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
            accuracy=(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
            accuracies.append(accuracy)
            valid_indices = np.where(test_y[:, i] != -1)[0]
            if len(valid_indices) > 0:
                auc = roc_auc_score(test_y[valid_indices, i], masked_predictions[valid_indices, i])
                auc_scores.append(auc)
            else:
                # Handle the case where there are no valid instances for the label
                auc_scores.append(np.nan)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            print('Acc',true_positive,false_positive,false_negative,true_negative)
        # Evaluate the model using the evaluate method and return loss and accuracy
        # Compute mean metrics across all labels
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)
        mean_auc_score = np.mean(auc_scores)
        mean_accuracy=np.mean(accuracies)
        result=[mean_accuracy,mean_precision,mean_recall,mean_f1_score,mean_auc_score]
        final_result=[result,accuracies,precisions,recalls,f1_scores,auc_scores]
        return final_result
    
    def load_model(cls, filepath):
        """
        Load a saved model from an HDF5 file.
        
        Parameters:
            filepath (str): File path to load the model from.
        
        Returns:
            MultiInputNN2: An instance of the MultiInputNN2 class with the loaded model.
        """
        
        # Define custom objects dictionary with the custom loss function
        custom_objects = {'masked_binary_crossentropy': masked_binary_crossentropy}
        
        # Load the saved model from the specified filepath using the custom_objects dictionary
        loaded_model = tf.keras.models.load_model(filepath, custom_objects=custom_objects, compile=True)
        
        # # Initialize a new instance of the MultiInputNN2 class
        # # For demonstration purposes, I'm creating a new instance without initializing data_dict, cnode, pnode, etc.
        # # You may need to modify this part based on your actual implementation and requirements
        # loaded_instance = cls(data_dict={}, cnode=10, pnode=100, gnode=10, gonode=100, drop_rate=0.5)
        # loaded_instance.model = loaded_model  # Assign the loaded model to the instance's model attribute
        
        # # print(f"Model loaded from {filepath}")
        
        return loaded_model
    

    def cal_stage_accuracy(self, predictions, test_y):
        ''' calculate the stage-specific accuracy'''

        # Predict on test data
        # predictions = self.model.predict(test_x)
        
        # Apply the mask on the predictions
        masked_predictions = np.where(test_y == -1, -1, (predictions > 0.5).astype(np.int64))
        
        # Calculate accuracy
        # accuracy = np.mean(np.equal(masked_predictions, test_y),axis=0)
        # mean_accuracy = np.mean(np.equal(masked_predictions, test_y))
        # Initialize lists to store metrics for each label
        precisions = []
        recalls = []
        f1_scores = []
        auc_scores = []  # Initialize list to store AUC scores
        accuracies=[]

        # Calculate metrics for each label
        for i in range(test_y.shape[1]):  # Assuming test_y.shape[1] gives the number of labels/columns
            true_positive = np.sum((masked_predictions[:, i] == 1) & (test_y[:, i] == 1))
            false_positive = np.sum((masked_predictions[:, i] == 1) & (test_y[:, i] == 0))
            false_negative = np.sum((masked_predictions[:, i] == 0) & (test_y[:, i] == 1))
            true_negative = np.sum((masked_predictions[:, i] == 0) & (test_y[:, i] == 0))  


            precision = true_positive / (true_positive + false_positive + 1e-9)
            recall = true_positive / (true_positive + false_negative + 1e-9)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
            valid_indices = np.where(test_y[:, i] != -1)[0]
            if len(valid_indices) > 0:
                auc = roc_auc_score(test_y[valid_indices, i], masked_predictions[valid_indices, i])
                auc_scores.append(auc)
            else:
                # Handle the case where there are no valid instances for the label
                auc_scores.append(np.nan)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            accuracy=(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
            accuracies.append(accuracy)
        # Evaluate the model using the evaluate method and return loss and accuracy
        # Compute mean metrics across all labels
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)
        mean_auc_score = np.mean(auc_scores)
        mean_accuracy=np.mean(accuracies)
        result=[mean_accuracy,mean_precision,mean_recall,mean_f1_score,mean_auc_score]
        final_result=[result,accuracies,precisions,recalls,f1_scores,auc_scores]
        return final_result

    def save(self, filename):
        """
        Save the model to a single HDF5 file.
        
        Parameters:
            filepath (str): File path to save the model.
        """
        # # Create directory if it doesn't exist
        # directory = os.path.dirname(filepath)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        
        # Save the model
        self.model.save(filename)
        

    def summary(self):
        self.model.summary()












