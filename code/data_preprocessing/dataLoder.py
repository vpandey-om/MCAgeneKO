"""
This script is used to combine data such as single cell and bulk gene expression for MCA  in one DataFrame.
Gene ontology and pathways data 
Protein based Embedding data
"""

# Import necessary libraries

import os
import pandas as pd 
from geneExp import DataUtils,PhenoUtils
import pickle
import numpy as np 
import h5py
import re

# Initialize DataUtils and PhenoUtils to get gene expression and phenotype data
du=DataUtils()
scdf=du.getMCAdata()
bulkdf=du.getBulkdata('bulk_file')
pu=PhenoUtils()

# get input and output for blood stage 
phenodf=pu.scaled_pheno_data.copy()

# Filter and process the blood stage phenotype data

blood_columns=['Gene ID','Gene description','Blood RGR','Blood SD','Blood phenotype']
blood_data_columns=['Blood RGR','Blood SD']
xx=phenodf[blood_data_columns]
yy=phenodf[blood_columns]
pheno_data=yy[~xx.isna().any(axis=1)]

# Filter bulk data based on available genes in the blood stage phenotype data
xx=bulkdf[bulkdf['gene_x'].isin(pheno_data['Gene ID'].to_list())]
bulk_blood=xx[~xx.duplicated(subset='gene_x',keep='first')]

# Filter single-cell gene expression data based on available genes in the blood stage phenotype data
scdf_blood=scdf[scdf['gene_id'].isin(pheno_data['Gene ID'].to_list())]

# Normalizing and preparing the data for further processing


# get bulk data samples 
# ap2ko=du.organized_ap2ko_samples_shortname
stage_samples = [item for sublist in du.stage_samples for item in sublist]
all_bulk_samples=list(set(du.ap2ko_sample['Sample_title'].to_list()+stage_samples)&set(bulk_blood.columns.to_list()))
ap2timeNormalData=du.ap2time.copy()
ap2timeNormalData[du.ap2time_sample.file]=ap2timeNormalData[du.ap2time_sample.file]/ap2timeNormalData[du.ap2time_sample.file].sum(axis=0)
ap2timeNormalData = ap2timeNormalData.drop('id', axis=1)
final_df=pd.DataFrame(columns=['gene','gene_des','rgr','rgr_sd','pheno','scInput','bulkInput','ap2timeInput'])
index=0

# Getting the parent directory name
# where the current directory is present.
current = os.path.dirname(os.path.realpath(__file__))

geneko = os.path.dirname(current)


class DataLoad():

    # Define a class `DataLoad` to handle the consolidation and processing of various datasets

    def __init__(self):
        # Initialize the class with necessary file paths and load initial gene data
        self.inputFile=os.path.join(geneko,'data','InputTrainInfo.pickle')
        # get genes form gene features 
        orig_df=pd.read_csv(os.path.join(geneko,'data','GenesByText_Summary.txt'),sep='\t')
        orig_df=orig_df[['Gene ID','Product Description','Gene Name or Symbol','Previous ID(s)']].copy()
        # Remove duplicated rows based on the 'Name' column
        self.gene_df = orig_df.drop_duplicates(subset='Gene ID')

    def add_gene_exp_data(self):
        '''Add gene expression data, single-cell data, and bulk data to the gene dataframe'''
        ## add pheno type data 
        self.cdf=self.gene_df.copy()
        self.cdf=self.cdf.merge(pheno_data,on='Gene ID', how='left')
        self.cdf['pheno']=self.cdf['Blood phenotype'].copy()
        ## fill NA for nan and add flag=1
        self.cdf['pheno']=self.cdf['pheno'].fillna('NA')
        replace_col={'Fast':'Dispensable','Slow':'Dispensable'}
        self.cdf['pheno']=self.cdf['pheno'].replace(replace_col)
        self.cdf['pheno_bool']=1
        self.cdf.loc[self.cdf['pheno'].isin(['NA','Insufficient data']),'pheno_bool']=0

        ### get single cell gene expression data 
        self.cdf=self.cdf.merge(scdf,left_on='Gene ID',right_on='gene_id', how='left')
        ### if we get Nan values get put in the flag 
        self.cdf['sc_bool']=1
        self.cdf.loc[self.cdf[du.mca_sample.sample_id].isnull().any(axis=1),'sc_bool']=0
        column_means=self.cdf[du.mca_sample.sample_id].mean(axis=0)
        # Fill NaN values in selected columns with their respective means
        self.cdf[du.mca_sample.sample_id] = self.cdf[du.mca_sample.sample_id].fillna(column_means)

        ## add bulk data 
        self.cdf=self.cdf.merge(bulkdf,left_on='Gene ID',right_on='gene_x', how='left')
        self.cdf['bulk_bool']=1
        self.cdf.loc[self.cdf[all_bulk_samples].isnull().any(axis=1),'bulk_bool']=0
        column_means=self.cdf[all_bulk_samples].mean(axis=0)
        # Fill NaN values in selected columns with their respective means
        self.cdf[all_bulk_samples] = self.cdf[all_bulk_samples].fillna(column_means)

        ## add ap2time bulk data
        ## add bulk data 
        
        self.cdf=self.cdf.merge(ap2timeNormalData,left_on='Gene ID',right_on='geneId', how='left')
        self.cdf['ap2Gtime_bool']=1
        self.cdf.loc[self.cdf[du.ap2time_sample.file].isnull().any(axis=1),'ap2Gtime_bool']=0
        column_means=self.cdf[du.ap2time_sample.file].mean(axis=0)
        # Fill NaN values in selected columns with their respective means
        self.cdf[du.ap2time_sample.file] = self.cdf[du.ap2time_sample.file].fillna(column_means)

        self.samples=[du.mca_sample.sample_id,all_bulk_samples,du.ap2time_sample.file]
        self.bool_col=['sc_bool','bulk_bool','ap2Gtime_bool']
        self.additional=[du.mca_sample,du.ap2time_sample,du.ap2ko_sample,du.stage_samples]

        print("finished gene expression add ")

    def addPIFeatures(self):
        ''' Add protein-protein interaction features to the gene dataframe '''
        uniprot_gene_df=pd.read_csv(os.path.join(geneko,'data','uniprot_to_pbanka.csv'),sep=',',header=None)
        featurePI=pd.read_csv(os.path.join(geneko,'data','protein_features.txt'),sep=',')
        tdf=self.cdf.copy()
        df = featurePI.drop(columns=['Unnamed: 0'])
        features=df.loc[:,'degree':'laplacian_centrality'].columns
        
        merged_df=df.merge(uniprot_gene_df,left_on='node',right_on=0,how='left')
        tdf=tdf.merge(merged_df,left_on='Gene ID',right_on=1,how='left')
        tdf['pifeature_bool']=1
        tdf.loc[tdf[features].isnull().any(axis=1),'pifeature_bool']=0
        column_means=tdf[features].mean(axis=0)
        # Fill NaN values in selected columns with their respective means
        tdf[features] = tdf[features].fillna(column_means)
        self.cdf=tdf.copy()
        
        self.samples.append(features)
        self.bool_col.append('pifeature_bool')
        print("finished adding protein interaction features ")

    def addGeneFeatures(self):
        ''' Add gene features to the gene dataframe '''
        tdf=self.cdf.copy()
        df=pd.read_csv(os.path.join(geneko,'data','geneFeatures.txt'),sep='\t')
        df['genes']=df['genes'].str.replace('.1','')
        df['genes']=df['genes'].str.replace('.2','')

        features=df.loc[:,'entropy_2NT':].columns
        tdf=tdf.merge(df,left_on='Gene ID',right_on='genes',how='left')
        tdf['genefeature_bool']=1
        tdf.loc[tdf[features].isnull().any(axis=1),'genefeature_bool']=0
        column_means=tdf[features].mean(axis=0)
        # Fill NaN values in selected columns with their respective means
        tdf[features] = tdf[features].fillna(column_means)
        self.cdf=tdf.copy()
        self.samples.append(features)
        self.bool_col.append('pgenefeature_bool')
        print("finished gene features ")

    def addProteinEmbedding(self):
        ''' Add protein embedding features to the gene dataframe'''
        data = h5py.File(os.path.join(geneko,'data','PBANKA_protein_embeddings.h5'), 'r')
        proteins=list(data.keys())
        tdf=self.cdf.copy()
        m=len(proteins)
        n=len(data[proteins[0]])
        arr = np.empty((m,n,))
        arr[:] = np.nan
        pbankaIds={}
        
        for i,protein in enumerate(proteins):
            xx=re.findall(r"gene=(.*?)\s\|",protein)
            pbankaIds[xx[0]]=data[protein][:]
            arr[i,:]=data[protein]
        df=pd.DataFrame(data=pbankaIds)
        df=df.T
        custom_cols = ['embeding_' + str(i) for i in df.columns]

        df.columns = custom_cols
        features=df.columns
        df_reset = df.reset_index()
        tdf=tdf.merge(df_reset,left_on='Gene ID',right_on='index',how='left')
        tdf['protein_embedding_bool']=1
        tdf.loc[tdf[features].isnull().any(axis=1),'protein_embedding_bool']=0
        column_means=tdf[features].mean(axis=0)
        # Fill NaN values in selected columns with their respective means
        tdf[features] = tdf[features].fillna(column_means)
        self.cdf=tdf.copy()
        self.samples.append(features)
        self.bool_col.append('protein_embedding_bool')
        print("finished protein embedding features ")
        
    def addAp2time(self):
        '''# Add Ap2time cluster data to the gene dataframe'''

        df=pd.read_excel(os.path.join(geneko,'data','ap2timeCluster.xlsx'))
        df['Gene_ID']=df['Gene_ID'].str.replace('.1','')
        df['Gene_ID']=df['Gene_ID'].str.replace('.2','')
        df['Gene_ID']=df['Gene_ID'].str.replace(':rRNA','')
        
        ###
        one_hot_encoded = pd.get_dummies(df['Cluster'], prefix='clust_ap2time').astype(int)
    

        # Concatenate the one-hot encoded DataFrame with the original DataFrame
        df = pd.concat([df, one_hot_encoded], axis=1)
        tdf=self.cdf.copy() 
        tdf['ap2gtime_bool']=1
        features=df.loc[:,'0h':'Log2 M/F '].columns 
        features_clust=df.loc[:,'clust_ap2time_1':].columns
        tdf=tdf.merge(df,left_on='Gene ID',right_on='Gene_ID',how='left')
        
        # Perform one-hot encoding on the 'Name' column
        fixInClust=df[features_clust].sum(axis=0).idxmax() #### fix in this cluster
        tdf.loc[tdf[features].isnull().any(axis=1),'ap2gtime_bool']=0
        column_means=tdf[features].mean(axis=0)
        ##
        ##
        # Fill NaN values in selected columns with their respective means
        tdf[features] = tdf[features].fillna(column_means)
        tdf[features_clust] = tdf[features_clust].fillna(0)
        self.cdf=tdf.copy()
        self.samples.append(features)
        self.samples.append(features_clust)
        self.bool_col.append('ap2gtime_bool')
        print("finished ap2g time ")

    def addAp2ko(self):
        ''' Add Ap2ko cluster data to the gene dataframe'''
        df=pd.read_excel(os.path.join(geneko,'data','ap2koCluster.xlsx'))
        df['Gene_ID_new']=df['Gene_ID_new'].fillna('NA')
        ###
        one_hot_encoded = pd.get_dummies(df['Cluster'], prefix='clust_ap2ko').astype(int)
    

        # Concatenate the one-hot encoded DataFrame with the original DataFrame
        df = pd.concat([df, one_hot_encoded], axis=1)
        
        
        
        tdf=self.cdf.copy() 
        tdf['ap2gko_bool']=1
        features_clust=df.loc[:,'clust_ap2ko_1':].columns
        tdf=tdf.merge(df,left_on='Gene ID',right_on='Gene_ID_new',how='left')
        tdf.loc[tdf[features_clust].isnull().any(axis=1),'ap2gko_bool']=0
        
        tdf[features_clust] = tdf[features_clust].fillna(0)
        self.cdf=tdf.copy()
        self.samples.append(features_clust)
        self.bool_col.append('ap2gko_bool')
        print("finished ap2g ko ")

    def addGOterms(self):
        ''''  Add Gene Ontology (GO) terms to the gene dataframe '''
        tdf=self.cdf.copy() 
        gene_go=pickle.load(open(os.path.join(geneko,'data','gene_to_GO.pickle'),'rb'))
        goterms=[]
        count=0
        for index in tdf.index:
            gene=tdf.loc[index,'Gene ID']
            tmp=gene_go[gene_go['genes']==gene]
            if tmp.empty:
                goterms.append(['GO:0016021|0']) ### by default zeros
                count=count+1
            else:
                goterms.append(tmp['go_anot'].to_list()[0])
        print(count)
        tdf['goterm']=goterms

        go_anot=gene_go['go_anot'].to_list()
        flat_list = [item.split('|')[0] for sublist in go_anot for item in sublist]
        goTotal=list(set(flat_list))
        gos_dict={}
        ####
        for i,item in enumerate(goTotal):
            gos_dict[item]=i
        ### again get int vector

        go_score=[]
        go_bool=[]
        
        for index in tdf.index:
            arr = np.zeros(( len(gos_dict)), dtype=np.float32)
            if not tdf.loc[index,'goterm']=='NA':
                go_bool.append(1)
                for item in tdf.loc[index,'goterm']:
                    t_id, score = item.split('|')
                    if t_id in gos_dict:
                        arr[gos_dict[t_id]] = float(score)
                 
            else:
                go_bool.append(0)

            go_score.append(arr)
        tdf['go_score']=go_score
        tdf['go_bool']=go_bool
        self.cdf=tdf.copy()
    
        

        
        
        
        
# Instantiate the dataLoad class and execute methods to consolidate and process data
dl=DataLoad()
dl.add_gene_exp_data()
dl.addPIFeatures()
dl.addGeneFeatures()
dl.addProteinEmbedding()
dl.addAp2time()
dl.addAp2ko()
dl.addGOterms()

# remove duplicates 
cdf=dl.cdf.drop_duplicates(subset='Gene ID')
# Save the consolidated dataframe and related information to a pickle file
pickle.dump([cdf,dl.samples,dl.bool_col,dl.additional],open(os.path.join(geneko,'data','allData_final.pickle'),'wb'))
