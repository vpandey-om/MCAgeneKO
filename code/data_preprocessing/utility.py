# Import necessary libraries
import pickle 
import plotnine as pn
import pandas.api.types as pdtypes
import pandas as pd 
from scipy import stats
import  numpy as np
from statsmodels.stats.weightstats import ztest

# Define the Utils class
class Utils():

    def __init__(self):
       # Define the different stages
        self.stage=['Ring','Trophozoite','Schizont','Male gametocyte','Female gametocyte',
        'Ookinete','oocyst 7d','oocyst 10d','Sporozoite','EEF_24h','EEF_48h','EEF_54h','EEF_60h']

        # Load data from pickle files and CSV files

        allData=pickle.load(open('/Users/vikash/gitlab/genekoai/data/allData_final.pickle','rb'))
        self.stage_sample=pd.read_csv('/Users/vikash/gitlab/genekoai/data/stage_samples_anotations.txt',sep='\t')
        self.df=allData[0]
        self.list_samples=allData[1]
        self.list_boolcol=allData[2]
        self.sample_detail=allData[3]
        self.stage2=['blood','Male gametocyte','Female gametocyte',
        'Ookinete','oocyst 7d','oocyst 10d','Sporozoite','liver']
        # single cell data stage
        self.single_cell_stage=['Merozoite', 'Ring', 'Trophozoite','Schizont', 'Male','Female_gametocyte', 'Ookinete',
                            'Oocyst','Gland_sporozite','Injected_sporozite','Liver_stage' ]
        
        y_all=np.ones((self.df.shape[0],), dtype=int)*-1
        y_all[self.df.loc[:,'pheno_bool']==1]=0
        y_all[self.df.loc[:,'pheno']=='Essential']=1
        self.df['y_all']=y_all

        ## Gene ontology and some important groups from bushell et al paper
        apico=pd.read_csv('/Users/vikash/gitlab/genekoai/data/Apicoplast.txt',on_bad_lines='warn',sep='\t')
        mitochondria=pd.read_csv('/Users/vikash/gitlab/genekoai/data/Mitochondria.csv',on_bad_lines='warn')
        res=pickle.load(open('/Users/vikash/gitlab/genekoai/data/genegopf.pickle','rb'))
        genes=res[0]
        go_genes=res[1]
        enriched_go=pd.read_excel('/Users/vikash/gitlab/genekoai/data/enriched_go.xlsx',sheet_name='Enrichment p-values')
        go_genes['apico']=apico['current_version_ID'].to_list()
        go_genes['mito']=mitochondria['current_version_ID'].to_list()
        
        # Extract unique values from both columns
        # rows = list(set(value for values in go_genes.values() for value in values))
        rows=self.df['Gene ID'].to_list()
        df=self.df.copy()
        cols = [k for k in go_genes.keys()]
        df.loc[:,cols]=0
        df['go_bool']=0
        # for col in cols:
        #     df[col] = 0
        # binary_matrix = np.zeros((1, len(cols)), dtype=int)[0]
        
        for i in df.index:
            flag=0
            for j, col_key in enumerate(cols):
                if df.loc[i,'Gene ID'] in go_genes.get(col_key, []):
                    # binary_matrix[ j] = 1
                    df.loc[i,col_key]=1
                    flag=1
            if flag==1:
                df.loc[i,'go_bool']=1

        
        # godf=pd.DataFrame(columns=cols,data=binary_matrix)
        # # godf['y_all']=y
        # # godf['Gene ID']=rows
    
        # self.df = pd.concat([self.df, godf], axis=1)
        self.go_features=cols
        self.df=df.copy()
        # apico=pd.read_excel('/Users/vikash/gitlab/genekoai/data/pheno_catogory.xlsx',sheet_name='Apicoplast')
        # mito=pd.read_excel('/Users/vikash/gitlab/genekoai/data/pheno_catogory.xlsx',sheet_name='Mitochondrion')
        # genego=pd.read_csv('/Users/vikash/gitlab/genekoai/data/geneid2gopbnew.csv',sep='\t')
        # genego=pd.read_csv('/Users/vikash/gitlab/genekoai/data/geneid2gopbnew.csv',sep='\t')
        # genego[['ID','GO','ID,GO']]=genego['ID,GO'].str.split(',',n=-1,expand=True)
        # ### get previous to new ids 
        # prev_new={}
        # anot_df=pd.read_csv('/Users/vikash/gitlab/GenesByText_Summary_62.txt',sep='\t')
        # anot_df=anot_df.fillna('NA')
        # import pdb;pdb.set_trace()
        # for item in genego['ID'].unique():
        #     xx=anot_df[anot_df['Previous ID(s)'].str.contains(item)]
        #     if not xx.empty:
        #         prev_new[item]=xx['Gene ID'].to_list()[0]
        #     else:
        #         print(item)

        
        # go=pd.read_csv('/Users/vikash/gitlab/genekoai/data/GenesByText_GOTerms_63.txt',sep='\t')
        # go=go.fillna('NA')
        
        
    def select_aggregate_features(self):
        '''''Select aggregate features based on different stages.'''
        ### get stage wise data 
        df=self.df.copy()
        ### single cell data 
        stages=['Trophozoite','Oocyst','Liver_stage' ]
        
        
        features=[]
        for i,st in enumerate(stages):
            tmp=self.sample_detail[0][self.sample_detail[0].ShortenedLifeStage3==st]
            cols=tmp['sample_id'].to_list()
            common_cols=list(set(cols)&set(df.columns))
            if len(common_cols)>0:
                medianvals=df[common_cols].median(axis=1)
                madvals=stats.median_abs_deviation(df[common_cols].values,axis=1)
                f1='sc_median'+st
                f2='sc_mad'+st
                f3='percent_exp_cells'+st
                df[f1]=medianvals
                features.append(f1)
                df[f2]=madvals
                features.append(f2)
                ## find number of expressed cells 
                # Count columns where values are less than the cutoff value row-wise
                column_counts = df[common_cols].apply(lambda row: (row < 0.1).sum(), axis=1)
                df[f3] =column_counts / len(common_cols)
                features.append(f3)
        ## bulk data 
        stages=['Trophozoite','EEF_24h']
        for i,st in enumerate(stages):
            tmp=self.stage_sample[self.stage_sample['stages']==st]
            cols=tmp['samples'].to_list()
            common_cols=list(set(cols)&set(df.columns))
            if len(common_cols)>0:
                medianvals=df[common_cols].median(axis=1)
                madvals=stats.median_abs_deviation(df[common_cols].values,axis=1)
                f1='bulk_median'+st
                f2='bulk_mad'+st
                df[f1]=medianvals
                features.append(f1)
                df[f2]=madvals
                features.append(f2)

        ### embedding 
        common_cols=self.list_samples[5]
        medianvals=df[common_cols].median(axis=1)
        madvals=stats.median_abs_deviation(df[common_cols].values,axis=1)
        # f1='protein_embedding_median'
        # f1='protein_embedding_mad'
        # df[f1]=medianvals
        # df[f2]=madvals
        # features.append(f1)
        # features.append(f2)
        for item in self.list_samples[5]:
            features.append(item)
        ## merge with other phenotype data 
        self.merge_df_and_pheno(df)
        return df,features
                
            
            
    def merge_df_and_pheno(self,df):
        '''Merge phenotype data with the dataframe.''' 
        pheno_df=pd.read_excel('/Users/vikash/gitlab/genekoai/data/phenotype_data.xlsx')
        ### self pheno types 
        ## merge all the data 
        merged_df=df.merge(pheno_df,on='Gene ID',how='left')
        merged_df.loc[:,'liver_pheno']=0
        merged_df.loc[merged_df['Liver phenotype']=='Reduced','liver_pheno']=1
        merged_df.loc[:,'male_pheno']=0
        merged_df.loc[merged_df['Male fertility phenotype']=='Reduced','male_pheno']=1
        merged_df.loc[:,'female_pheno']=0
        merged_df.loc[merged_df['Female fertility phenotype']=='Reduced','female_pheno']=1
        merged_df.loc[:,'oocyst_pheno']=0
        merged_df.loc[(merged_df['Female fertility phenotype']=='Reduced')|(merged_df['Oocyst phenotype']=='Reduced'),'oocyst_pheno']=1
        merged_df.loc[:,'sporo_pheno']=0
        merged_df.loc[merged_df['Sporozoite phenotype']=='Reduced','sporo_pheno']=1
        self.merged_df=merged_df.copy()
       

    def plot_protein_embedding(self):
        ''' Plot protein embedding for visualization. '''
        df=self.df.copy()
        df=df.loc[~(df['y_all']==-1),:]
        common_cols=self.list_samples[5]
        viz_df=pd.DataFrame()
        medianvals=df[common_cols].median(axis=1)
        madvals=stats.median_abs_deviation(df[common_cols].values,axis=1)
        viz_df['median_rpm']=medianvals
        viz_df['mad']=madvals
        viz_df['phenotype']=df['y_all'].values
        viz_df['stages']=['embedding']*len(medianvals)
        res=ztest(viz_df[viz_df.phenotype==0]['median_rpm'].dropna().values, viz_df[viz_df.phenotype==1]['median_rpm'].dropna().values)
        res1=ztest(viz_df[viz_df.phenotype==0]['mad'].dropna().values, viz_df[viz_df.phenotype==1]['mad'].dropna().values)
        print(res,res1)
        return viz_df
    
    def plot_stage_specific_sc_data(self):
        ''' Plot stage-specific blood stage data '''  
        ldfs=[]
        df=self.df.copy()
        df=df.loc[~(df['y_all']==-1),:]
        for i,st in enumerate(self.single_cell_stage):
            viz_df=pd.DataFrame()
            
            tmp=self.sample_detail[0][self.sample_detail[0].ShortenedLifeStage3==st]
            cols=tmp['sample_id'].to_list()
            common_cols=list(set(cols)&set(df.columns))
            if len(common_cols)>0:
                meanvals=np.log2(df[common_cols].mean(axis=1))
                stdvals=df[common_cols].std(axis=1)
                medianvals=np.log2(df[common_cols].median(axis=1))
                madvals=np.log2(stats.median_abs_deviation(df[common_cols].values,axis=1))
                viz_df['mean_rpm']=meanvals
                viz_df['sd']=stdvals
                viz_df['median_rpm']=medianvals
                viz_df['mad']=madvals
                ## find number of expressed cells 
                # Count columns where values are less than the cutoff value row-wise
                column_counts = df[common_cols].apply(lambda row: (row < 0.1).sum(), axis=1)
                viz_df['Percentexpressed_cells'] =column_counts / len(common_cols)
                viz_df['phenotype']=df['y_all'].values
                viz_df['stages']=[st]*len(meanvals)
                viz_df = viz_df.replace([np.inf, -np.inf], np.nan)
                ldfs.append(viz_df)
                res=ztest(viz_df[viz_df.phenotype==0]['median_rpm'].dropna().values, viz_df[viz_df.phenotype==1]['median_rpm'].dropna().values)
                res1=ztest(viz_df[viz_df.phenotype==0]['mad'].dropna().values, viz_df[viz_df.phenotype==1]['mad'].dropna().values)
                res3=ztest(viz_df[viz_df.phenotype==0]['Percentexpressed_cells'].dropna().values, viz_df[viz_df.phenotype==1]['Percentexpressed_cells'].dropna().values)
                print(st,res,res1,res3)
            else:
                print('Not present',cols)
        viz_df_all=pd.concat(ldfs,axis=0)
        return viz_df_all
    


    def plot_stage_specific_blood_stage_data(self):
        ''' select blood stage-specific data '''
        ldfs=[]
        df=self.df.copy()
        df=df.loc[~(df['y_all']==-1),:]
        for i,st in enumerate(self.stage):
            viz_df=pd.DataFrame()
            tmp=self.stage_sample[self.stage_sample['stages']==st]
            cols=tmp['samples'].to_list()
            common_cols=list(set(cols)&set(df.columns))
            if len(common_cols)>0:
                meanvals=np.log2(df[common_cols].mean(axis=1))
                stdvals=df[common_cols].std(axis=1)
                medianvals=np.log2(df[common_cols].median(axis=1)+0.01)
                madvals=np.log2(stats.median_abs_deviation(df[common_cols].values,axis=1)+0.00001)
                viz_df['mean_rpm']=meanvals
                viz_df['sd']=stdvals
                viz_df['median_rpm']=medianvals
                viz_df['mad']=madvals
                viz_df['phenotype']=df['y_all'].values
                viz_df['stages']=[st]*len(meanvals)
                ldfs.append(viz_df)
                res=ztest(viz_df[viz_df.phenotype==0]['median_rpm'].dropna().values, viz_df[viz_df.phenotype==1]['median_rpm'].dropna().values)
                res1=ztest(viz_df[viz_df.phenotype==0]['mad'].dropna().values, viz_df[viz_df.phenotype==1]['mad'].dropna().values)
                print(st,res,res1)
            else:
                print('Not present',cols)
        viz_df_all=pd.concat(ldfs,axis=0)
        return viz_df_all
            


    def plot_violin(self,df,stages,filepdf,xlab,xlegend):
        '''Plot violin plots for different stages.'''
        color_gam_dict={1:'#ca0020',0:'#0571b0'}
        # df['Pathway'] = df['Paths'].astype(pdtypes.CategoricalDtype(categories=orderPath))
        # df['Phenotype'] = df['Phenotype'].astype(pdtypes.CategoricalDtype(categories=['Reduced','Not reduced']))
        # df['Sex'] = df['Sexes'].astype(pdtypes.CategoricalDtype(categories=['Female','Male']))
        
        
        ###
        for item in stages:
            if item not in df['stages'].unique():
                stages.remove(item)
        orderStage=stages
        df['stages'] = df['stages'].astype(pdtypes.CategoricalDtype(categories=orderStage))
        df['phenotype'] = df['phenotype'].astype(pdtypes.CategoricalDtype(categories=[1, 0]))
        plot=(pn.ggplot(df, pn.aes(y=xlab, x="stages",fill='phenotype'))
            
        + pn.coord_flip()
        # + pn.geom_violin(df,style='full',position='dodge')
        + pn.geom_violin(df, style='left-right',draw_quantiles=0.5)
        + pn.geom_point(pn.aes(color='phenotype'),size=0.75)
        #  + pn.geom_boxplot(width=0.4,alpha=0.7,size=0.65,show_legend=False)
        + pn.scale_color_manual(values=color_gam_dict,labels={1:'Essential',0:'Not essential'})
        # Add custom annotations to specific facets
        #  + pn.geom_text( pn.aes(x='order+0.3', y='y', label='label'), data=custom_pvals,size=6, color='red')
        + pn.labs(color = 'Phenotype',x='Stages', y=xlegend,fill='Phenotype')
        + pn.scale_fill_manual(values=color_gam_dict,labels={1:'Essential',0:'Not essential'})
        # + pn.annotate('text', y=max(df['Fertility rate'])+0.5, x=df["Pathway"], label='NAAAAAA')
        #  + pn.facet_wrap('Sex')

        + pn.theme(
                # figure_size=(11, 4.8), ### 4.8
                # legend_direction="vertical",
                # legend_box_spacing=0.4,
                # legend_position='none',
                axis_line=pn.element_line(size=1, colour="black"),
                # panel_grid_major=pn.element_line(colour="#d3d3d3"),
                panel_grid_major=pn.element_blank(),
                panel_grid_minor=pn.element_blank(),
                panel_border=pn.element_blank(),
                panel_background=pn.element_blank(),
                # plot_title=pn.element_text(size=15, family="Arial",
                #                         face="bold"),
                text=pn.element_text(family="Arial", size=11),
                axis_text_x=pn.element_text(colour="black", size=10),
                axis_text_y=pn.element_text(colour="black", size=10),
            )
        )


        # Set the figure size


        plot.save(filepdf, dpi=300)



        