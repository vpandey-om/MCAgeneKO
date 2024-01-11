
#This script seems to be dealing with a data processing pipeline, 
#Import Necessary Modules:
from utility import Utils
from sklearn.preprocessing import MinMaxScaler
import pickle
#Initialize Utility Class and Extract Features:
ut=Utils()


df,features=ut.select_aggregate_features()
 
#Copy Merged DataFrame:
merged_df=ut.merged_df.copy()
# scaler = MinMaxScaler()
# X=scaler.fit_transform(concatenated_array)

#Define Lists of Features:
# Various lists of features are defined, such as ex_features, pe_features, gene_features, etc.
tmp=ut.sample_detail[0]
cells=cols=tmp['sample_id'].to_list()
ex_features=features[:13]
pe_features=features[13:]
gene_features=['entropy_2NT', 'entropy_3NT', 'AA_MI', 'AT_MI', 'AC_MI', 
               'AG_MI', 'TA_MI', 'TT_MI', 'TC_MI', 'TG_MI', 'CA_MI', 'CT_MI', 'CC_MI', 
               'CG_MI', 'GA_MI', 'GT_MI', 'GC_MI', 'GG_MI', 'sum_MI', 'AAA_CMI', 'AAT_CMI', 'AAC_CMI', 
               'AAG_CMI', 'ATA_CMI', 'ATT_CMI', 'ATC_CMI', 'ATG_CMI', 'ACA_CMI', 'ACT_CMI', 'ACC_CMI', 'ACG_CMI', 
               'AGA_CMI', 'AGT_CMI', 'AGC_CMI', 'AGG_CMI', 'TAA_CMI', 'TAT_CMI', 'TAC_CMI', 'TAG_CMI', 'TTA_CMI', 'TTT_CMI',
                 'TTC_CMI', 'TTG_CMI', 'TCA_CMI', 'TCT_CMI', 'TCC_CMI', 'TCG_CMI', 'TGA_CMI', 'TGT_CMI', 'TGC_CMI', 'TGG_CMI',
                   'CAA_CMI', 'CAT_CMI', 'CAC_CMI', 'CAG_CMI', 'CTA_CMI', 'CTT_CMI', 'CTC_CMI', 'CTG_CMI', 'CCA_CMI', 'CCT_CMI', 
                   'CCC_CMI', 'CCG_CMI', 'CGA_CMI', 'CGT_CMI', 'CGC_CMI', 'CGG_CMI', 'GAA_CMI', 'GAT_CMI', 'GAC_CMI', 'GAG_CMI',
                     'GTA_CMI', 'GTT_CMI', 'GTC_CMI', 'GTG_CMI', 'GCA_CMI', 'GCT_CMI', 'GCC_CMI', 'GCG_CMI', 'GGA_CMI', 'GGT_CMI', 
                     'GGC_CMI', 'GGG_CMI', 'sum_CMI', 'kld_single', 'kld_di', 'kld_tri']

### 
#all_features=cells+ex_features+pe_features+gene_features+ut.go_features
all_features=cells+pe_features+gene_features+ut.go_features
y_var=['y_all','liver_pheno','male_pheno','female_pheno','sporo_pheno','oocyst_pheno']
stages=tmp.ShortenedLifeStage3.unique()

# Create a dictionary to store unique values and their indices
stages_indices = {}

# Iterate over unique values and find their indices
for value in stages:
    indices = tmp[tmp.ShortenedLifeStage3 == value].index.tolist()
    stages_indices[value] = indices

# Filter Data Based on Conditions:

boolIdx=(~(merged_df['y_all']==-1))&(merged_df['go_bool']==1)&(merged_df['protein_embedding_bool']==1) & (merged_df['genefeature_bool']==1)
# df_red=df.loc[boolIdx,:]

concatenated_array=merged_df[all_features].copy()

#Scale Data Features:

scaler = MinMaxScaler()
X=scaler.fit_transform(concatenated_array)
## get protein_embedding_bool 
X_red=X[boolIdx,:]

y_red=merged_df.loc[boolIdx,y_var]

# Create a Dictionary to Store Data Outputs:
# Various data components are stored in this dictionary for further analysis or visualization.
data_output={}
data_output['df']=merged_df
data_output['reduced_X']=X_red
data_output['X']=X
data_output['reduced_Y']=y_red
data_output['stages output']=y_var
data_output['stages cells index']=stages_indices
data_output['utility']=ut
data_output['cells']=cells
data_output['protein_features']=pe_features
data_output['gene_features']=gene_features
data_output['gene_ontology_features']=ut.go_features


# Path to save the pickle file
pickle_file_path = 'data/data_output.pkl'

# Save the data_output dictionary to a pickle file
with open(pickle_file_path, 'wb') as f:
    pickle.dump(data_output, f)

print(f"Data output has been saved to {pickle_file_path}")






