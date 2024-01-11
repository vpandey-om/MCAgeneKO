# MCAgeneKO
This repository covers the modeling, data analysis, and visualization techniques for predicting gene knockouts in malaria, aiding in the identification of potential candidates for drug discovery.

## Overview
Our initiative aims to promote drug discovery in the context of malaria parasite research by combining machine and deep learning techniques to decipher gene functions. Specifically, we used prediction algorithms to conduct a genome-wide search for critical genes required by malaria parasites. These models were carefully trained using data from Plasmodium berghei knockout mutant growth rates. We added gene and protein-centric features to improve the model's prediction capabilities. Our systematic efforts resulted in a noteworthy accuracy rate of 70% in predicting cases where specific genes are considered essential based only on gene-related traits.

## Prerequisites

Users need to install before using the Snakemake workflow.

- Python (>=3.7)

## Data Collection and Preprocessing 

Data for phenotypes were gathered from several research that examined genetic knockouts in Plasmodium berghei at various life cycle stages. As previously stated, a comprehensive genome-scale knockout investigation [Blood screen](https://pubmed.ncbi.nlm.nih.gov/28708996/) was conducted to better understand gene involvement throughout the blood stage. Because continuous drug selection for knockouts is only possible during the parasite's asexual blood phases, this strategy aids in the research of genes important in the parasite's later stages post-blood. Following research focused on the liver stage [Liver screen](https://pubmed.ncbi.nlm.nih.gov/31730853/), while another bar-seq screen investigation found genes important for male and female gametocyte development. In addition, as stated in [Fetility screen](https://www.biorxiv.org/content/10.1101/2023.12.25.572958v1.abstract), a screening was performed to identify genes associated with male and female fertility. The phenotypic web interface compiles all of these screening datasets for easy access.

Single-cell gene expression data originate from the Malaria Cell Atlas ([MCA](https://www.malariacellatlas.org/)) and bulk-RNA gene expression data were collected from multiple studies, including [Study1](https://pubmed.ncbi.nlm.nih.gov/30177743/) and [Study2](https://pubmed.ncbi.nlm.nih.gov/28081440/). Additionally, bulk-RNA gene expression data are from diffrent life cycle stages of malaria parasites.

The preprocessing scripts can be found in the data_preprocessing directory within the code folder. To obtain all the necessary data, execute the following command:
~~~
python datautil.py
~~~

Furthermore, I've provided comprehensive data for training, validation, and testing encapsulated in a pickle file named "data_output.pkl", which resides in the data folder.

## Modeling
The malaria parasite undergoes a complex life cycle involving both vertebrate hosts and mosquito vectors ([detail](https://www.malariavaccine.org/malaria-and-vaccines/vaccine-development/life-cycle-malaria-parasite)). The phenotype data are avilable for six distinct stages:Blood,Liver,Male Gametocyte,Female Gametocyte,Oocyst,Sporozoite.

We approached this challenge as a multilabel classification task, where each life stage is associated with phenotype data categorized as either present (yes) or absent (no).
![Multilabel classification](https://github.com/vpandey-om/MCAgeneKO/blob/main/Figures/multilable.png)

### Challenges in Classification Tasks:
1. For genes associated with the blood phenotype, data for the remaining five stages (Liver, Male Gametocyte, Female Gametocyte, Oocyst, Sporozoite) are unavailable due to experimental constraints.
2. There's a significant imbalance between the 'yes' and 'no' classes for the mentioned five stages, leading to issues with data imbalance and missing values.
#### Addressing Classification Challenges
To address the difficulties of our classification problems, we used a two-pronged approach: the creation of a bespoke adaptive loss function and the incorporation of a bagging technique.

1. Custom Adaptive Loss Function: To address the missing value data issues in our dataset, we created a custom adaptive loss function. This particular function includes a masking mechanism that ensures loss computations are only performed on available data points. This adaptive strategy improves the model's resilience and accuracy in the presence of incomplete data by focusing on the data at hand.

2. Bagging Technique: To supplement our unique loss function, we used the bagging technique, a durable and effective ensemble method. This strategy requires training numerous models on different subsets of the dataset and then combining their predictions. Bagging successfully mitigates variation, improves model stability, and improves our classification framework's overall predictive capabilities by using the collective knowledge of several models.

### Challenges in Classification Tasks:


