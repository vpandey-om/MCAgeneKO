"""
This script interacts with a PostgreSQL database to fetch and preprocess gene expression, single-cell data, and phenotype data.
The script is structured into two utility classes: PhenoUtils for phenotype-related data and DataUtils for general data management.

PhenoUtils Class:
- Connects to the database and fetches phenotype data along with gene annotations.
- Performs log2 transformations and merges the phenotype data with gene annotations.
- Provides functionalities to scale phenotype data for different stages.

DataUtils Class:
- Manages various data tables from the database such as ap2ko, ap2time, eefdata, etc.
- Defines methods for calculating counts per million (CPM) and fetching bulk and MCA-specific data.
- Organizes data samples and stages for better management and analysis.

Key Attributes:
- DATABASE_URI: PostgreSQL database connection URI.
- Various data tables are loaded using SQLAlchemy's pandas read_sql_query method.
- Phenotype and gene annotations are processed, merged, and organized into meaningful structures.

"""

# Importing necessary libraries

from sqlalchemy import create_engine
import pandas as pd
from functools import reduce
from sqlalchemy.sql import text

from sqlalchemy import inspect
# from plasmogem import pu
import numpy as np


# Database connection details

POSTGRES = {
    'user': 'vikash',
    'pw': '****',
    'db': 'phenodb',
    'host': 'localhost',
    'port': '5432',
}



# Constructing DATABASE_URI 
DATABASE_URI='postgresql+psycopg2://%(user)s:%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES


class PhenoUtils:
    """
    Utility class for handling phenotype-related data.
    """
    def __init__(self):
        """Initialize database connection and load phenotype and gene annotation data."""

        self.engine = create_engine(DATABASE_URI) 
        self.phenodata_df = pd.read_sql_query('''select * from phenotable''',con=self.engine)
        ### log2 transformation for blood stage
        # self.phenodata_df['blood_SD']=self.phenodata_df['blood_SD']/(self.phenodata_df['blood_RGR']*np.log(2))
        # self.phenodata_df['blood_SD']=self.phenodata_df['blood_SD'].apply(lambda x: np.log2(x) if(np.all(pd.notnull(x))) else x)
        self.phenodata_df['blood_RGR_main']=self.phenodata_df['blood_RGR']+self.phenodata_df['blood_SD']
        self.phenodata_df['blood_RGR']=self.phenodata_df['blood_RGR'].apply(lambda x: np.log2(x) if(np.all(pd.notnull(x))) else x)
        self.phenodata_df['blood_RGR_main']=self.phenodata_df['blood_RGR_main'].apply(lambda x: np.log2(x) if(np.all(pd.notnull(x))) else x)
        self.phenodata_df['blood_SD']=self.phenodata_df['blood_RGR_main']-self.phenodata_df['blood_RGR']
        self.geneanot_df = pd.read_sql_query('''select * from geneanot''',con=self.engine)
        ##### add gene description in the table
        df=self.phenodata_df.merge(self.geneanot_df, left_on='new_pbanka_id', right_on='new_id',how='left')
        df=df.drop_duplicates(subset=['new_pbanka_id'], keep='first')
        self.phenodata_df['Gene description']=df['description'].to_list()
        # initilize data
        self.empty_df=pd.DataFrame()

        # This is the dictionary which can be used to give a reasonable name for data columns
        self.pheno_col_rename={'B2toMG_RGR':'Oocyst conversion rate', 'B2toMG_SD':'Oocyst SD',
        'B2toMG_pheno': 'Oocyst phenotype', 'MGtoSG_RGR':'Sporozoite conversion rate',
        'MGtoSG_SD':'Sporozoite SD','MGtoSG_pheno':'Sporozoite phenotype', 'SGtoB2_RGR':'Liver conversion rate',
        'SGtoB2_SD':'Liver SD', 'SGtoB2_pheno': 'Liver phenotype', 'blood_RGR':'Blood RGR',
        'blood_SD':'Blood SD', 'blood_pheno':'Blood phenotype', 'female_fertility_RGR': 'Female fertility rate',
        'female_fertility_SD' :'Female fertility SD' , 'female_fertility_pheno':'Female fertility phenotype',
        'male_fertility_RGR' :'Male fertility rate','male_fertility_SD':'Male fertility SD',
        'male_fertility_pheno':'Male fertility phenotype', 'female_gam_RGR':'Female gametocyte conversion rate',
        'female_gam_SD':'Female gametocyte SD', 'female_gam_pheno':'Female gametocyte phenotype',
        'male_gam_RGR':'Male gametocyte conversion rate', 'male_gam_SD':'Male gametocyte SD',
        'male_gam_pheno':'Male gametocyte phenotype','new_pbanka_id':'Gene ID'}

        # This is arranged based on malaria life cycle stages order
        self.arranged_pheno_col=['Gene ID','Gene description','Blood RGR','Blood SD','Blood phenotype','Female gametocyte conversion rate',
        'Female gametocyte SD','Female gametocyte phenotype','Male gametocyte conversion rate',
        'Male gametocyte SD','Male gametocyte phenotype','Female fertility rate','Female fertility SD',
        'Female fertility phenotype','Male fertility rate','Male fertility SD','Male fertility phenotype',
        'Oocyst conversion rate','Oocyst SD','Oocyst phenotype','Sporozoite conversion rate','Sporozoite SD',
        'Sporozoite phenotype','Liver conversion rate','Liver SD','Liver phenotype']

        self.stage_pheno=['Blood','Female Gametocyte','Male Gametocyte','Female fertility',
        'Male Fertility','Oocyst','Sporozoite','Liver']

        self.pheno_rate_columns=['Blood RGR','Female gametocyte conversion rate','Male gametocyte conversion rate',
        'Female fertility rate','Male fertility rate','Oocyst conversion rate','Sporozoite conversion rate',
        'Liver conversion rate']

        self.pheno_sd_columns=['Blood SD','Female gametocyte SD','Male gametocyte SD','Female fertility SD',
        'Male fertility SD','Oocyst SD','Sporozoite SD','Liver SD']

        self.pheno_call_columns=['Blood phenotype','Female gametocyte phenotype','Male gametocyte phenotype',
        'Female fertility phenotype','Male fertility phenotype','Oocyst phenotype','Sporozoite phenotype',
        'Liver phenotype']
        # Used colors to show different kind of phenotypes 
        self.change_color_all_pheno={'Slow':'#fdae61', 'Dispensable':'#abd9e9', 'Fast':'#2c7bb6', 'Essential':'#d7191c',
        'Insufficient data':'#737373','Reduced':'#fc8d59','Not reduced':'#91bfdb','No power':'#737373','No data':'#737373',np.nan:'#737373','NA':'#737373'}

        # Scale diffrent kind of data
        self.scaled_pheno_data=self.scalePhenoData()


    def scalePhenoData(self):
            ''' scale the phenotype data for diffrent stages  '''

            filtered_df=self.phenodata_df.copy()
            filtered_df=filtered_df.rename(columns=self.pheno_col_rename)
            filtered_df=filtered_df[self.arranged_pheno_col].copy()
            ###
            scaled_df=filtered_df.copy()
            for i,col in enumerate(self.pheno_rate_columns):
                maxi=scaled_df[col].max()
                mini=scaled_df[col].min()
                scaled_df[col]=scaled_df[col].apply(lambda x: (x-mini)/(maxi-mini) if(np.all(pd.notnull(x))) else x)

                #### compute new SD

                scaled_df['max']=filtered_df[col]+filtered_df[self.pheno_sd_columns[i]]
                # maxi=scaled_df['max'].max()
                # mini=scaled_df['max'].min()
                scaled_df['max']=scaled_df['max'].apply(lambda x: (x-mini)/(maxi-mini) if(np.all(pd.notnull(x))) else x)
            
            
                scaled_df[self.pheno_sd_columns[i]]=scaled_df['max']-scaled_df[col]
                # print(scaled_df[self.pheno_sd_columns[i]].min(),scaled_df[self.pheno_sd_columns[i]].max())
            return scaled_df

    

class DataUtils:
    """
    Utility class for managing and organizing general data operations.
    """
    def __init__(self):
        """Initialize database connection and load various data tables."""
        # get phenotpe dataframe from database
        self.engine = create_engine(DATABASE_URI)
        # inspector = inspect( self.engine)

        # Get table information
        # print(inspector.get_table_names())

        # result_set = self.engine.execute("SELECT * FROM geneanot")  
        # for r in result_set:  
        #      print(r)
       
        self.ap2ko = pd.read_sql_query(text('''select * from ap2koexp'''),con=self.engine)
       
        self.ap2ko_sample = pd.read_sql_query(text('''select * from ap2koexp_sample'''),con=self.engine)
        self.ap2time = pd.read_sql_query('''select * from ap2time''',con=self.engine)
   
        self.ap2time_sample = pd.read_sql_query('''select * from ap2time_sample''',con=self.engine)
 
        self.eef = pd.read_sql_query('''select * from eefdata''',con=self.engine)
        self.eef_sample = pd.read_sql_query('''select * from eefdata_sample''',con=self.engine)
        self.mca_sample = pd.read_sql_query('''select * from mcasample''',con=self.engine)
        self.mca_umap = pd.read_sql_query('''select * from mcaumap''',con=self.engine)
        self.mfdata= pd.read_sql_query('''select * from mfdata''',con=self.engine)
        allItems= pd.read_sql_query('''select * from mcajsondata''',con=self.engine)
        # with self.engine.connect() as conn:
        #     allItems=conn.execute("SELECT * FROM mcajsondata").fetchall()
        try:
            scdata_df=pd.DataFrame.from_dict(allItems.loc[0,'data']) ### hard code
            self.scdata_df = scdata_df

        except:
            print('Eroor in loading mca data')
        self.geneanot = pd.read_sql_query('''select * from geneanot''',con=self.engine)
        self.empty_df=pd.DataFrame()
        # get stage specific data
        self.stage_name=['Ring','Trophozoite','Schizont','Gametocyte','Male gametocyte','Female gametocyte',
        'Ookinete','Sporozoite','EEF_24h','EEF_48h','EEF_54h','EEF_60h']

        self.stage_samples=[['EF_ring_4h_A', 'EF_ring_4h_B'],['EF_trophozoite_16h_A','EF_trophozoite_16h_B'],
        ['wt_S_1', 'wt_S_3','wt_S_4','wt_S_6','wt_S_2','wt_S_5'],['wt_G_3', 'wt_G_2','wt_G_1'],
        ['Male1', 'Male2', 'Male3'],['Female1','Female2', 'Female3'],['wt_O_1','wt_O_2','wt_O_3'],
        ['sporozoites_A','sporozoites_B'],['EEF_24h_A','EEF_24h_B'], ['EEF_48h_A', 'EEF_48h_B'],
        ['EEF_54h_A', 'EEF_54h_B'],['EEF_60h_A', 'EEF_60h_B']]

        self.organized_ap2ko_samples=['wt_P.berghei schizont culture','ap2-g ko_P.berghei schizont culture',
        'ap2-g2 ko_P.berghei schizont culture','ap2-o ko_P.berghei schizont culture',
        'ap2-o2 ko_P.berghei schizont culture','ap2-o3 ko_P.berghei schizont culture',
        'ap2-o4 ko_P.berghei schizont culture','ap2-sp ko_P.berghei schizont culture',
        'ap2-sp2 ko_P.berghei schizont culture','ap2-sp3 ko_P.berghei schizont culture',
        'ap2-l ko_P.berghei schizont culture','PBANKA_131970 ko_P.berghei schizont culture',
        'wt_purified P.berghei gametocytes','ap2-o ko_purified P.berghei gametocytes',
        'ap2-o2 ko_purified P.berghei gametocytes','ap2-o3 ko_purified P.berghei gametocytes',
        'wt_purified P.berghei ookinetes','ap2-o ko_purified P.berghei ookinetes',
        'ap2-o2 ko_purified P.berghei ookinetes','ap2-o3 ko_purified P.berghei ookinetes'
        ]
        self.organized_ap2ko_samples_shortname=['wt S','ap2-g S','ap2-g2 S','ap2-o S','ap2-o2 S',
        'ap2-o3 S','ap2-o4 S','ap2-sp S','ap2-sp2 S','ap2-sp3 S','ap2-l S','PBANKA_131970 S','wt G',
        'ap2-o G','ap2-o2 G','ap2-o3 G','wt O','ap2-o O','ap2-o2 O','ap2-o3 O']
        self.organized_ap2ko_samples_ticks=['wt','g','g2','o','o2',
        'o3','o4','sp','sp2','sp3','l','HC','wt',
        'o','o2','o3','wt','o','o2','o3']
        self.organized_ap2ko_samples_stages=['S','S','S','S','S',
        'S','S','S','S','S','S','S','G',
        'G','G','G','O','O','O','O']
        
        


    




    def getIntColumns(self,df):
        """Get integer columns from a dataframe."""

        bool = ~(df.dtypes==object)
        countcols=df.columns[bool].drop(['id','geneanot.id']) ## these are id columns
        return countcols

    def calculateCPM(self,df):
        """Calculate Counts Per Million (CPM) for a given dataframe."""
        countcols=self.getIntColumns(df)
        sumvals=df[countcols].sum(axis=0)
        tdf= df.loc[:,countcols].div(sumvals, axis=1)
        tdf=tdf*1e6
        df.loc[:,countcols]=tdf.loc[:,countcols].copy()
        return df


    def getBulkdata(self,bulk_file,ids=[]):
        """Fetch bulk data for specific IDs."""

        eef_df=self.calculateCPM(self.eef)
        ap2ko=self.calculateCPM(self.ap2ko)
        dfs = [eef_df, ap2ko, self.mfdata]
        bulk_df = reduce(lambda left,right: pd.merge(left,right,on='geneanot.id',how='outer'), dfs)
        if len(ids)>0:
            final_bulk_df=bulk_df[bulk_df['geneanot.id'].isin(ids)]
        else:
            final_bulk_df=bulk_df
        # import pdb;pdb.set_trace()
        # ## filter stage-specific data
        # final_bulk_df=self.stage_specific_bulk_data(final_bulk_df,bulk_file)
        return final_bulk_df

    def getMCAdata(self,ids=[]):
        """Organize and get Malaria cell atalas data."""
        if len(ids)>0:
            scdata_df=self.scdata_df[self.scdata_df['geneanot.id'].isin(ids)]
        else:
            scdata_df=self.scdata_df
        return scdata_df


    def stage_specific_bulk_data(self,df,bulk_file):
        """Organize stage-specific bulk data."""
        ## try to find bulk data at all stages of MCA cycle
        df1=pd.DataFrame(index=df.index,columns=['gene_x'])
        print('combining all stages data')
        for i,stage in enumerate(self.stage_name):
            df1[stage+'_mean_bulk']=df[self.stage_samples[i]].mean(axis = 1, skipna = True)
            df1[stage+'_std_bulk']=df[self.stage_samples[i]].std(axis = 1, skipna = True)
        df1.to_csv(bulk_file,sep='\t')
        return df






