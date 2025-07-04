# C.V. Cocuzza, 2025. Symptom fingerprinting code for "Brain network dynamics reflect psychiatric illness status and transdiagnostic symptom profiles across health and disease". Cocuzza et al., 2025. 

# The steps of this pipeline are organized into python functions that are possible to adapt to other datasets, 
# but the TCP dataset is quite unique (at the time of preparing this manuscript) in that it collected many more 
# behavioral, cognitive, and clinical measures (i.e., self-report surveys, cognitive batteries, clinical instruments, etc.) 
# than is typical. Thus, we've uploaded (to GitHub) de-identified behavioral data files to make implementing the functions here easier 
# (and hopefully easier to adapt to other datasets). The participant IDs match those openly available on OpenNeuro and NDA for 
# TCP dataset, but all PHI has been removed. Example usage is recommended throughout along with explanatory notes.

# NOTE: the data in all_TCP_behav_concatenated_mapped_domains.pkl was pre-labelled and verified by coauthors 
# (see Methods of manuscript; and large Supplemental Table that also reflects this data)

# WORKFLOW (corresponds to modular functions below; so see each function for more detailed notes): 
# 1) Organize and curate data: fingerprints_curate_Cocuzza()
# 2) Transform measures toward Gaussianity: fingerprints_transform_Cocuzza(), cnd_project_imputation_R_pipe.Rmd
# ** NOTE: this step is multi-part and partially involves RStudio. See cnd_project_imputation_R_pipe.Rmd
# 3) Binarize 0-inflated variables: fingerprints_binarize_Cocuzza()
# 4) Impute missing values/scores: fingerprints_imputation_Cocuzza()
# 5) Estimate distance matrix (individual differences correlations of normalized and transformed measures): fingerprints_distance_Cocuzza()
# 6) Input distance matrix into agglomerative hiearchical clustering to partition data into empirically-driven dimensions of functioning: fingerprints_clustering_Cocuzza()
# ** NOTE: this step partially involves RStudio, see cnd_project_clustering_checks_pipe.Rmd
# 7) Perform PCA on each cluster of measures (normalized and imputed): fingerprints_PCA_Cocuzza()

# NOTE: steps 8 and 9 are TBA to GitHub. 
# 8) Use PCA results (i.e., weight by factor loadings) and pre-labellings to name clusters: fingerprints_cluster_naming_Cocuzza()
# 9) Quantify multi-dimensional symptom profiles (i.e., symptom fingerprints) for each participant: fingerprints_profiles_Cocuzza()

# NOTE: we were able to take many of the steps here *because* of the Transdiagnostic Connectome Project (TCP) dataset 
# having a relatively *very large* set of behavioral, cognitive, and clinical measures to use. That is to say, 
# if we excluded a given measure, it was very likely that there were still other measures to cover that proportion of variance in functioning. 
# Even with all the above steps, the final set of behavioral data included 110 variables. 
# This may not be the case (and is likely not) for other datasets, thus these steps should be adapted with caution and thorough consideration.

'''
# Example code to load behavioral data shared in GitHub repo 
# (copy and paste this section into jupyter notebook, python script, etc.)
# dataHere_All can be used in first function below; see notes for all functions for usages (these are modular and in-order).

import pandas as pd 
import numpy as np 
import pickle

dirHere = '/directory/path/to/where/you/saved/scripts/and/data/' # CHANGE

keysAll = np.load(dirHere + 'all_TCP_behav_concatenated_keys.npy',allow_pickle=True).copy()

dataHere_All = pd.read_csv(dirHere + 'all_TCP_behav_concatenated.csv')

with open(dirHere + 'all_TCP_behav_concatenated_mapped_domains.pkl', 'rb') as f:
    behavTestLegend = pickle.load(f)

numMeasures = dataHere_All.shape[1]-1 # 1st column is subject ID 
print(f"Initial data dimensions: N = {dataHere_All.shape[0]} participants and {numMeasures} behavioral/clinical/cognitive measures.")
'''

# Other notes: 

# Using missForest to impute missing values in observed data 
# NOTE: can use missCompare in R to see if missForest is still best solution (TBA), but for now going straight to missForest 
# See here: https://github.com/rawat126/Feature-Engineering/blob/c523d48154d7808ee26214e3c70066a0981d634a/Miss%20Forest%20Imputation/Miss_forest_imputation.ipynb#L6
# And here: https://ragvenderrawat.medium.com/miss-forest-imputaion-the-best-way-to-handle-missing-data-feature-engineering-techniques-2e6922e5cecb
# Also see: https://scikit-learn.org/stable/modules/impute.html
# Also see: https://ragvenderrawat.medium.com/miss-forest-imputaion-the-best-way-to-handle-missing-data-feature-engineering-techniques-2e6922e5cecb
# See: https://www.analyticsvidhya.com/blog/2022/05/handling-missing-values-with-random-forest/

####################################################################################
# IMPORTS (some may need to be installed) 
import sys
import os
#import subprocess
#import time

import numpy as np

import pandas as pd
import pickle 
import h5py as h5

import scipy
#import scipy.io as spio
import scipy.stats as stats
#from scipy import signal
import sklearn 
import statistics as stat 

from sklearn.ensemble import RandomForestClassifier # for categorical data 
from sklearn.ensemble import RandomForestRegressor # for continuous data 

####################################################################################
def fingerprints_curate_Cocuzza(dataHere_All,keysAll,saveResults=False,saveDir=None,percentThresh=25,removeRedunancies=True,removeSparseSubjs=False,percentThresh_Subjs=50):

    '''
    Data organizing & curating: trim measures based on a series of related conditions:
    a) remove measures with > threshold % of subjects missing 
    b) remove redundant meaures (e.g., having both t-scores and raw scores) -- this was manually coded for TCP and agreed upon by coauthors (and based on literature/best practices)
    c) remove subjects above a certain threshold of sparseness. This was done in the TCP data release manuscript but we did *not* do it here because downstream steps tended to exclude those anyway (see imputation function for how we accounted for this) 
    
    INPUTS:
        dataHere_All: see top of script for how to load example data; from csv file with all behavioral data concatenated into dataframe 
        keysAll: see top of script for how to load example data; from npy file with all keys for dataHere_All
        saveResults: Boolean w/ Default False; whether or not to also save out the curated dataframe (into .csv)
        saveDir: if saveResults=True; a directory path (include entire string) to save out results (csv) from curation steps 
        percentThresh: subject-missingness threshold for measure removal, Default is 25
        removeRedunancies: Boolean w/ Default True; whether or not to remove redundant measures (recommended) 
        removeSparseSubjs: Boolean w/ Default False; whether or not to remove sparse subjects (not recommended for this study) 
        percentThresh_Subjs: If removeSparseSubjs is True (not recommended), this is the sparsity threshold to use. 50% (Default) is relatively lenient. 

    OUTPUTS: 
        dataHere_TrimFinal: new dataframe with measures trimmed; returned and also saved to [saveDir + 'dataHere_TrimFinal_BeforeGaussTest.csv'] when saveResults=True
    '''

    # a) Remove measures with more than a given % of subjects missing 

    numMeasures = dataHere_All.shape[1]-1 # ignore 1st column, which is subject ID 
    allKeys = dataHere_All.keys().to_numpy().copy()
    
    # Loop over measures and find those with >percentThresh% of subjects' missing data 
    measuresToRemove = []
    measuresToRemove_Ixs = []
    for measureIx in range(numMeasures):
        thisMeasureStr = keysAll[1:][measureIx] # skip 1st column (subject ID)

        measureVec_AllSubjs = dataHere_All[thisMeasureStr].to_numpy().copy()
        numMissingSubjs = np.where(np.isnan(measureVec_AllSubjs))[0].shape[0]
        percentMissingSubjs = np.floor((numMissingSubjs / dataHere_All.shape[0]) * 100)
        
        if percentMissingSubjs > percentThresh:
            print(f"{thisMeasureStr} has {percentMissingSubjs}% subjects missing, flagging it for removal...")
            measuresToRemove.append(thisMeasureStr)
            measuresToRemove_Ixs.append(measureIx)
            
    measuresToRemove = np.asarray(measuresToRemove)
    measuresToRemove_Ixs = np.asarray(measuresToRemove_Ixs)
    
    dataHere_Trimmed = dataHere_All.copy()
    dataHere_Trimmed = dataHere_Trimmed.drop(measuresToRemove,axis=1)

    keys_Trimmed = allKeys[1:].copy()
    keys_Trimmed = np.delete(keys_Trimmed,measuresToRemove_Ixs)

    print(f"\nTrimming the dataset by {measuresToRemove.shape[0]} behavioral/clinical/cognitive measures that had > {percentThresh}% of subjects missing. "+
          f"New number of measures = {dataHere_Trimmed.shape[1]-1}.\n")

    # b) data curating continued from above: trim redundant variables

    # NOTE: important to avoid results being overly driven by 1 scale; but I'm making it optional 
    # NOTE: these were decided by hand because it typically involved reviewing the available literature and finding the field standard/preference for that measure 

    # NOTES on what was trimmed: 
    # POMS, SHIPLEY: keep t-scores 
    # YMRS: keep sub-scales and remove total (sub-scales likely offer more diverse coverage of behavioral/cognitive constructs / explain more variance)
    # BAPQ, BIS, MSPSS, TCI, POMS, COGFQ, RSRI: keep sub-scales and remove total
    # MADRS: keep total and remove subscales because the subscales are 0 inflated and would end up being binarized (which isn't ideal) later anyway
    # LRIFT: keep total and remove subscales because the subscales are 0 inflated and would end up being binarized (which isn't ideal) later anyway
    # PUM: keep total-via-average and remove total-via-sum
    # CTQ: removing to keep as a demographic variable/covariate (childhood trauma questionnaire is unique in that it's based on past experiences) 
    # CRT: these variables end up being binarized (not ideal) and overfit the clustering solutions downstream
    
    dataHere_Trimmed_NonRedundant = dataHere_Trimmed.copy()
    if removeRedunancies:
        # This list corresponds to the notes above (re: "what was trimmed" and why)
        keysToRM = ['lrift_employment','lrift_household', 'lrift_school', 'lrift_work','lrift_spouse_relate', 'lrift_children_relate','lrift_relatives_relate', 
                    'lrift_friends_relate','lrift_relationships', 'lrift_satisfaction', 'lrift_recreation',
                    'madrs_reduced_appetite', 'madrs_apparent_sadness','madrs_concentration_issues', 'madrs_inability_feel','madrs_lassitude', 
                    'madrs_pessimism', 'madrs_reported_sadness','madrs_reduced_sleep', 'madrs_suicidal_thoughts','madrs_inner_tension',
                    'ctq_denial_validity', 'ctq_emo_abuse', 'ctq_emo_neglect','ctq_phys_abuse', 'ctq_phys_neglect', 'ctq_sex_abuse',
                    'POMS_anger','POMS_confusion','POMS_depression','POMS_fatigue','POMS_tension','POMS_vigour','POMS_total_raw',
                    'SHIPLEY_vocab','ymrs_total','bapq_total_avg','bis_total_sum','mspss_total','tci_total_sum','cogfq_total',
                    'crt_correct_first_3','crt_intuitive_first_3', 'crt_correct_all_5','crt_intuitive_all_5','rsri_total_average','pum_total_sum']

        dataHere_Trimmed_NonRedundant = dataHere_Trimmed_NonRedundant.drop(keysToRM,axis=1)
        print(f"Removing {len(keysToRM)} behavioral/clinical/cognitive measures that were redundant and/or circular with other measures. "+
              f"New number of measures = {dataHere_Trimmed_NonRedundant.shape[1]-1}.\n")
    
    # c) remove subjects with > a certain % of missing variables 
    
    # NOTE: for the purposes of CND project we did NOT do this, but keeping here for other projects/uses/datasets. Reasons:
    # (1) subjects will be removed later anyway (only certain functional runs were of-interest for NMF/dynamics pipeline) 
    # (2) we need as many subjects as possible for train / test / validation splits,
    # (3) so instead we opted for a well-validated imputation procedure (see later function) 
    
    dataHere_TrimFinal = dataHere_Trimmed_NonRedundant.copy() 

    if removeSparseSubjs:
        
        subjKeysAll = dataHere_Trimmed_NonRedundant['subjectkey'].to_numpy().copy()
        keysHere = dataHere_Trimmed_NonRedundant.keys().to_numpy().copy()
        
        subjsToRemove_IDs = []
        subjsToRemove_Ixs = []
        for subjIx in range(dataHere_Trimmed_NonRedundant.shape[0]):
            subjID = subjKeysAll[subjIx]
            subjVec = dataHere_Trimmed_NonRedundant[dataHere_Trimmed_NonRedundant['subjectkey']==subjID][keysHere[1:]].values[0,:].copy()
            numNaNs = np.where(np.isnan(subjVec))[0].shape[0]
            percNaNs = np.floor(((numNaNs/subjVec.shape[0])*100))
            if percNaNs > percentThresh_Subjs:
                subjsToRemove_IDs.append(subjID)
                subjsToRemove_Ixs.append(subjIx)
                print(f"Flagging subject {subjID} for having {percNaNs}% missing behavioral/clinical/cognitive measures...")

        subjsToRemove_IDs = np.asarray(subjsToRemove_IDs)
        subjsToRemove_Ixs = np.asarray(subjsToRemove_Ixs)

        dataHere_TrimFinal = dataHere_TrimFinal.drop(subjsToRemove_Ixs,axis=0)

        print(f"\nTrimming the dataset by {subjsToRemove_Ixs.shape[0]} subjects that were missing > {percentThresh_Subjs}% of behavioral/clinical/cognitive measures. "+
              f"New number of subjects = {dataHere_TrimFinal.shape[0]}.\n")
        
        # QA:
        subjIxVec_All = np.arange(dataHere_Trimmed_NonRedundant.shape[0])
        subjIxVec_All = np.delete(subjIxVec_All,subjsToRemove_Ixs)
        if not np.all(subjKeysAll[subjIxVec_All] == dataHere_TrimFinal['subjectkey'].to_numpy()):
            print(f"WARNING: subject ID order is off, please check.")
            
    elif not removeSparseSubjs:
        print(f"\nNOT trimming the dataset by subjects missing behavioral/clinical/cognitive measures. Number of subjects = {dataHere_TrimFinal.shape[0]}.\n")

    # SAVE
    if saveResults:
        dataHere_TrimFinal.to_csv(saveDir + 'dataHere_TrimFinal_BeforeGaussTest.csv')
    
    return dataHere_TrimFinal
    
####################################################################################
def fingerprints_transform_Cocuzza(dataHere_TrimFinal,normData=False,useZ=False,saveResults=False,saveDir=None):
    '''
    The goal of this step is to normalize variables that are deemed non-Gaussian. NOTE that this was aided by an RStudio toolkit called bestNormalize, with the following workflow:
    
    1) Run fingerprints_curate_Cocuzza() [function above] to curate which variables will be part of the final set for analyses.
    2) Run those variables through this function fingerprints_transform_Cocuzza() to identify which need to be normalized --> save the identified non-Gaussian variables.
    3) Load non-Gaussian variables identified in step 2 into RStudio and run bestNormalize (see cnd_project_imputation_R_pipe.Rmd).
    4) Re-run this function fingerprints_transform_Cocuzza() to see if any of the variables are still being flagged as non-Gaussian (likely a small proportion of original set of variables)
    5) The variables that are still non-Gaussian will need to be manually inspected. The vast majority were 0-inflated (or otherwise point-inflated), in which case we binarized (see next function, fingerprints_binarize_Cocuzza()), but a small number were reasonably close to having a normal distribution and were kept as-is. It may also be reasonable to trim those last remaining variables from further analyses, depending on how problematic the distribution is for later steps (e.g., strongly multimodal distributions might need further consideration)
    
    INPUTS: 
        dataHere_TrimFinal: output dataframe of fingerprints_curate_Cocuzza(); NOTE if saveResults=True in fingerprints_curate_Cocuzza(), you could load this csv then use as input (or some variant thereof) 
        normData: Boolean w/ Default False. Whether or not to pre-normalize with z-score or min-max. 
        useZ: Boolean w/ Default False. Only when normData=True. If True, pre-normalize with z-score; if False, pre-normalize with min-max method. 
        saveResults: Boolean w/ Default False (but we recommend setting to True); whether or not to also save out the curated dataframe (into .csv)
        saveDir: if saveResults=True; a directory path (include entire string) to save out results (csv) from transformation steps 
        
    OUTPUTS: 
        dataHere_CND_ToTransform: a dataframe of the variables you should pass to R for transformation; if you do a 2nd pass of this function (see notes above), the resulting dataframe from R will be the input dataHere_TrimFinal (make sure normData is False) 
        dataHere_TrimFinal_Normed: only if normData=True; an intermedite output dataframe; all the curated data before identifying which variables need to go to R for transforming; i.e., all data in the input dataHere_TrimFinal plus min-max normalization or z-scoring (based on useZ input)
        
    '''
    
    # Normalize data: optional 
    if normData:
        dataHere_TrimFinal_Normed = dataHere_TrimFinal.copy()

        if useZ: 
            # NOTE: indexed at 1 to skip the first column (subject IDs), but may need to adjust to 2 if another, extra index column was added in the last step
            for keyIx in range(1,dataHere_TrimFinal.keys().to_numpy().shape[0]): 
                keyHere = dataHere_TrimFinal.keys().to_numpy()[keyIx]
                vecHere = dataHere_TrimFinal[keyHere].to_numpy().copy()
                zVec = stats.zscore(vecHere,nan_policy='omit').copy()
                dataHere_TrimFinal_Normed[keyHere] = zVec.copy()

            # SAVE: 
            if saveResults:
                dataHere_TrimFinal_Normed.to_csv(saveDir + 'dataHere_TrimFinal_Normed_Z.csv')

        elif not useZ:
            # NOTE: indexed at 1 to skip the first column (subject IDs), but may need to adjust to 2 if another, extra index column was added in the last step
            for keyIx in range(1,dataHere_TrimFinal.keys().to_numpy().shape[0]):
                keyHere = dataHere_TrimFinal.keys().to_numpy()[keyIx]
                vecHere = dataHere_TrimFinal[keyHere].to_numpy().copy()
                normVec = minMaxNorm(vecHere,minVal=0,maxVal=1).copy()
                dataHere_TrimFinal_Normed[keyHere] = normVec.copy()

            # SAVE: 
            if saveResults:
                dataHere_TrimFinal_Normed.to_csv(saveDir + 'dataHere_TrimFinal_Normed_MinMax.csv')     

    elif not normData:
        # SAVE: 
        if saveResults:
            dataHere_TrimFinal.to_csv(saveDir + 'dataHere_TrimFinal_NotNormed.csv')


    # Identify non-gaussian data for transformation
    # Here, I'm running some tests to identify which variables need transforming, I'll then save those out --> go to R --> load them back in 
    # See cnd_project_imputation_R_pipe.rmd
    if normData: 
        keysHere_All = dataHere_TrimFinal_Normed.keys().to_numpy().copy()
        dataHere_TestGauss_Pass1 = dataHere_TrimFinal_Normed.copy()

    elif not normData:
        keysHere_All = dataHere_TrimFinal.keys().to_numpy().copy()
        dataHere_TestGauss_Pass1 = dataHere_TrimFinal.copy()
    
    # Test whether scale's distribution differs from normal distribution (D’Agostino’s K^2 test)
    keyIxs_NonNormal = []
    keyIxs_NonNormal_Cusp = [] # i.e., metrics that only just fail the test, so inspect further 
    for keyIx in range(1,keysHere_All.shape[0]):
        keyHere = keysHere_All[keyIx]
        dataVecHere = dataHere_TestGauss_Pass1[keyHere].to_numpy().copy()
        kStat,pVal = stats.normaltest(dataVecHere,nan_policy='omit')
        if pVal<0.05:
            #print(f"{keyHere} (py index {keyIx}) is likely not normal")
            keyIxs_NonNormal.append(keyIx)
            if pVal > 0.01: 
                print(f"{keyHere} (py index {keyIx}) may be on the cusp, please check")
                keyIxs_NonNormal_Cusp.append(keyIx)

    keyIxs_NonNormal = np.asarray(keyIxs_NonNormal)
    keyIxs_NonNormal_Cusp = np.asarray(keyIxs_NonNormal_Cusp)
    print(f"\nOf {keysHere_All.shape[0]-1} measures, {keyIxs_NonNormal.shape[0]} were identified as not matching normal distribution\n")

    # Test Gaussianity with Shapiro-Wilk test 
    keyIxs_ShapiroWilk = []
    keyIxs_ShapiroWilk_Cusp = [] # i.e., metrics that only just fail the test, so inspect further 
    for keyIx in range(1,keysHere_All.shape[0]):
        keyHere = keysHere_All[keyIx]
        dataVecHere = dataHere_TestGauss_Pass1[keyHere].to_numpy().copy()
        nanIxs = np.where(np.isnan(dataVecHere))[0]
        if nanIxs.shape[0]!=0:
            dataVecHere = np.delete(dataVecHere,nanIxs)
        kStat,pVal = stats.shapiro(dataVecHere)
        if pVal<0.05:
            #print(f"{keyHere} (py index {keyIx}) is likely not normal")
            keyIxs_ShapiroWilk.append(keyIx)
            if pVal > 0.01: 
                print(f"{keyHere} (py index {keyIx}) may be on the cusp, please check")
                keyIxs_ShapiroWilk_Cusp.append(keyIx)

    keyIxs_ShapiroWilk = np.asarray(keyIxs_ShapiroWilk)
    keyIxs_ShapiroWilk_Cusp = np.asarray(keyIxs_ShapiroWilk_Cusp)
    print(f"\nOf {keysHere_All.shape[0]-1} measures, {keyIxs_ShapiroWilk.shape[0]} were identified as non-Gaussian with Shapiro-Wilk\n")

    # Test normality with Anderson-Darling test 
    keyIxs_AD = []
    keyIxs_AD_Cusp = [] # i.e., metrics that only just fail the test, so inspect further 
    for keyIx in range(1,keysHere_All.shape[0]):
        keyHere = keysHere_All[keyIx]
        dataVecHere = dataHere_TestGauss_Pass1[keyHere].to_numpy().copy()
        nanIxs = np.where(np.isnan(dataVecHere))[0]
        if nanIxs.shape[0]!=0:
            dataVecHere = np.delete(dataVecHere,nanIxs)
        testModel = stats.anderson(dataVecHere,dist='norm')
        if testModel.statistic>testModel.critical_values[2]: # see testModel.significance_level for associated alphas for each critical value:
            keyIxs_AD.append(keyIx)
            if testModel.statistic<testModel.critical_values[4]: # see testModel.significance_level for associated alphas for each critical value 
                print(f"{keyHere} (py index {keyIx}) may be on the cusp, please check")
                keyIxs_AD_Cusp.append(keyIx)

    keyIxs_AD = np.asarray(keyIxs_AD)
    keyIxs_AD_Cusp = np.asarray(keyIxs_AD_Cusp)
    print(f"\nOf {keysHere_All.shape[0]-1} measures, {keyIxs_AD.shape[0]} were identified as non-normal with Anderson-Darling\n")

    # Test normality with Kolmogorov-Smirnov test 
    keyIxs_KS = []
    keyIxs_KS_Cusp = [] # i.e., metrics that only just fail the test, so inspect further 
    for keyIx in range(1,keysHere_All.shape[0]):
        keyHere = keysHere_All[keyIx]
        dataVecHere = dataHere_TestGauss_Pass1[keyHere].to_numpy().copy()
        nanIxs = np.where(np.isnan(dataVecHere))[0]
        if nanIxs.shape[0]!=0:
            dataVecHere = np.delete(dataVecHere,nanIxs)
        kStat,pVal = stats.kstest(dataVecHere,'norm')
        if pVal<0.05:
            #print(f"{keyHere} (py index {keyIx}) is likely not normal")
            keyIxs_KS.append(keyIx)
            if pVal > 0.01: 
                print(f"{keyHere} (py index {keyIx}) may be on the cusp, please check")
                keyIxs_KS_Cusp.append(keyIx)

    keyIxs_KS = np.asarray(keyIxs_KS)
    keyIxs_KS_Cusp = np.asarray(keyIxs_KS_Cusp)
    print(f"\nOf {keysHere_All.shape[0]-1} measures, {keyIxs_KS.shape[0]} were identified as non-normal with Kolmogorov-Smirnov\n")
    
    # Identify which measures/variables were identified across a few tests for normality/Gaussianity 
    uniqueVec,countsVec = np.unique(np.concatenate((keyIxs_NonNormal,keyIxs_ShapiroWilk,keyIxs_AD,keyIxs_KS)),return_counts=True)
    nonNormalWinnerIxs = np.where(countsVec>=2)[0].copy()
    nonNormalWinners = uniqueVec[nonNormalWinnerIxs]
    #print(f"\nAcross all 4 tests, {nonNormalWinners.shape[0]} measures were identified as "+
    #      f"non-normal for >=2 tests: \n{keysHere_All[nonNormalWinners]} \n(py indices: {nonNormalWinners})\n")
    print(f"\nAcross all 4 tests, {nonNormalWinners.shape[0]} measures were identified as non-normal for >=2 tests:\n")

    # And identify which metrics were on the cusp across a few tests 
    uniqueVec,countsVec = np.unique(np.concatenate((keyIxs_NonNormal_Cusp,keyIxs_ShapiroWilk_Cusp,keyIxs_AD_Cusp,keyIxs_KS_Cusp)),return_counts=True)
    cuspWinnerIxs = np.where(countsVec>=3)[0].copy() # lower number = less strict; higher number = more strict; max = 4 (b/c that's the number of tests I used) 
    if cuspWinnerIxs.shape[0]!=0:
        cuspWinners = uniqueVec[cuspWinnerIxs]
        print(f"\nAcross all 4 tests, the following {cuspWinners.shape[0]} measures were on the cusp "+
              f"for being non-normal for >=3 tests: \n{keysHere_All[cuspWinners]} \n(py indices: {cuspWinners})\n")

        cuspWinnerIxs_Adj = np.zeros_like(cuspWinnerIxs)
        for ixHere in range(cuspWinnerIxs.shape[0]):
            cuspWinnerIxs_Adj[ixHere] = np.where(nonNormalWinners==cuspWinners[ixHere])[0]

        # And generate the final list to send to R 
        nonNormalWinners = np.delete(nonNormalWinners,cuspWinnerIxs_Adj)
    keysToTransform = np.concatenate((np.asarray(['subjectkey']),keysHere_All[nonNormalWinners]))

    # Curate dataframe and save 
    dataHere_CND_ToTransform = dataHere_TestGauss_Pass1.copy()
    dataHere_CND_ToTransform = dataHere_CND_ToTransform[keysToTransform]
    
    if saveResults:
        if normData:
            if useZ:
                saveFile = saveDir + 'dataHere_TrimFinal_Normed_Z_ToTransform.csv'
            elif not useZ:
                saveFile = saveDir + 'dataHere_TrimFinal_Normed_MinMax_ToTransform.csv'
        elif not normData:
            saveFile = saveDir + 'dataHere_TrimFinal_NotNormed_ToTransform.csv'

        dataHere_CND_ToTransform.to_csv(saveFile)
        
    if normData:
        return dataHere_CND_ToTransform, dataHere_TrimFinal_Normed
    elif not normData: 
        return dataHere_CND_ToTransform

####################################################################################
def fingerprints_binarize_Cocuzza(dataHere_ToBinarize,normData=False,useZ=False,saveResults=False,saveDir=None):
    '''
    The goal of this step is to binarize measures that have been identified as 0-inflated (after 2 runs through fingerprints_transform_Cocuzza() above and RStudio's bestNormalize). 
    NOTE that there are a few possible ways to deal with point-inflated data (e.g., poisson transforms; gamma models; etc.), but we opted for binarization given that it's straightforward. 
    
    INPUTS:
        dataHere_ToBinarize: dataframe; where general formatting is the same as other functions, but the variables (columns) here are those identified for binarization
        normData: whether or not data was normed in prior transformation step/function; note this is only relevant for strings used to save results if saveResults=True
        useZ: if normData=True, whether or not data was normalized with z-scoring in prior transformation step/function; note this is only relevant for strings used to save results if saveResults=True
        saveResults: Boolean w/ Default False (but we recommend setting to True); whether or not to also save out the results dataframe (into .csv)
        saveDir: if saveResults=True; a directory path (include entire string) to save out results (csv) from binarization steps 
        
    OUTPUTS: 
        dataHere_Transformed_Binarized: dataframe with measures/variables binarized 
    '''
    
    # Binarize highly non-gaussians (0-inflated etc.): binarize based on mode  
    
    # Initialize dictionary to make new dataframe 
    dataHere_Transformed_Pass2_DICT = {'subjectkey':dataHere_ToBinarize['subjectkey'].to_numpy()}

    for metricIx in range(1,keysHere_All.shape[0]):
        thisKey = keysHere_All[metricIx]

        vecHere = dataHere_ToBinarize[thisKey].to_numpy()
        modeVal,countVal = stats.mode(vecHere)

        nanIxs = np.where(np.isnan(vecHere))[0] # hold out nans temporarily 
        vecHere[nanIxs] = 0

        modeIxs = np.where(np.round(vecHere,4)==np.round(modeVal,4))[0]
        vecHere_Bin = np.zeros((vecHere.shape[0]))
        vecHere_Bin[modeIxs] = 1

        vecHere_Bin[nanIxs] = np.nan # add nans back in 

        # NOTE: can use sklearn (see line below), but I think it does the opposite mapping from what we want (1's given to the non-mode)
        # vecHere_Bin = sklearn.preprocessing.binarize(vecHere.reshape(-1,1),threshold=modeVal)[:,0].copy()

        dataHere_Transformed_Pass2_DICT[thisKey] = vecHere_Bin.copy()

    dataHere_Transformed_Binarized = pd.DataFrame(dataHere_Transformed_Pass2_DICT).copy()
    
    # SAVE 
    if saveResults:
        if normData:
            if useZ:
                dataHere_Transformed_Binarized.to_csv(saveDir + 'dataHere_TrimFinal_Normed_Z_Transformed_Pass2.csv')
            elif not useZ:
                dataHere_Transformed_Binarized.to_csv(saveDir + 'dataHere_TrimFinal_Normed_MinMax_Transformed_Pass2.csv')
        elif not normData:
            dataHere_Transformed_Binarized.to_csv(saveDir + 'dataHere_TrimFinal_NotNormed_Transformed_Pass2.csv')

    return dataHere_Transformed_Binarized

####################################################################################
def fingerprints_imputation_Cocuzza(dataHere_Collated_ToImpute,percentTest=20,reNormData=True,useZ=False,saveResults=False,saveDir=None):
    
    '''
    The goal of this step is to impute missing data points (example: a given participant did not complete a survey) using the MissForest algorithm. See Chopra et al. 2025, Sci Data, for extensive testing on the TCP dataset, which suggested that missforest yielded the most robust imputation results. 
    NOTE: in the manuscript, we used a conservative train/test/validation split of the data. It is up to the user to divvy up their data appropriately.
    
    INPUTS:
        dataHere_Collated_ToImpute: this is a dataframe where all of the results of the prior steps (curating, transforming (twice), and binarizing) are all collated back together (i.e., the columns are stacked back into 1 dataframe)
        percentTest: percent of the data to set as the test set. Default is 20. e.g., if 20 is given, this is an 80/20 train/test split.
        reNormData: whether or not to re-normalize the data; the bestNormalize transformation process in R appears to put some (but not all) variables in a different space/range; and this is particularly important for min-max norming (i.e., putting data back into range of 0-1); Also should consider using this when prior steps/functions had normData=False. The idea here is that the prior transformations were on raw scores, then here normalize just before imputation.
        useZ: Boolean w/ Default False. Only when normData=True. If True, pre-normalize with z-score; if False, pre-normalize with min-max method. 
        saveResults: Boolean w/ Default False (but we recommend setting to True); whether or not to also save out the curated dataframe (into .csv)
        saveDir: if saveResults=True; a directory path (include entire string) to save out results (csv) from transformation steps 
    
    OUTPUTS: 
        dataHere_Imputed: dataframe where formatting matches inputted dataHere_Collated_ToImpute, but missing values have been imputed
    '''

    # Note: dataHere_Collated_ToImpute was pre-computed (and saved, if saveResults=True) above; user needs to collate those dataframes before using this function
    if reNormData:
        dataHere_ToImpute_Final = dataHere_Collated_ToImpute.copy()
        if not useZ:
            for keyIx in range(1,dataHere_Collated_ToImpute.keys().to_numpy().shape[0]):
                keyStrHere = dataHere_Collated_ToImpute.keys().to_numpy()[keyIx]
                vecHere = dataHere_Collated_ToImpute[keyStrHere].to_numpy().copy()
                vecHere_Normed = minMaxNorm(vecHere).copy()
                dataHere_ToImpute_Final[keyStrHere] = vecHere_Normed.copy()

    elif not reNormData:
        dataHere_ToImpute_Final = dataHere_Collated_ToImpute.copy()
    
    # MissForest with sklearn 

    subjIDs_DF = dataHere_ToImpute_Final['subjectkey'].to_numpy().copy()
    dataHere_ToImpute_Final_TrainTest = dataHere_ToImpute_Final.copy()
    nSubjsCND = dataHere_ToImpute_Final.shape[0]
    nSubjsTrainTest = dataHere_ToImpute_Final_TrainTest.shape[0]
    percTrainTest = (nSubjsTrainTest * 100)/nSubjsCND

    print(f"Final dataset for imputation: {nSubjsTrainTest} subjects (of N={nSubjsCND} total) by {dataHere_ToImpute_Final_TrainTest.shape[1]} measures. "+
          f"{percTrainTest}% subjects (n = {nSubjsTrainTest}) will be used for train/test")

    percentTrain = 100 - percentTest
    nSubjs_ImputePipe = dataHere_ToImpute_Final_TrainTest.shape[0]
    numTrain = int(np.round((percentTrain * nSubjs_ImputePipe)/100))
    numTest = int(np.round((percentTest * nSubjs_ImputePipe)/100))
    if numTest + numTrain != nSubjs_ImputePipe:
        print(f"train and test subject numbers do not equal full N, please check.")

    print(f"\nFor missForest: training set: {percentTrain}% of subjects (n = {numTrain}). Test set: {percentTest}% of subjects (n = {numTest}). (Total N={nSubjs_ImputePipe}).")
    
    # Loop over scales; NOTE: can loop over shuffles and take average (for continuous data) or mode (categorical data) 
    subjVec_Shuffled = np.arange(nSubjs_ImputePipe)
    np.random.shuffle(subjVec_Shuffled)
    subjsTrain = subjVec_Shuffled[:numTrain].copy()
    subjsTest = subjVec_Shuffled[numTrain:].copy()

    keysHere_All = dataHere_ToImpute_Final_TrainTest.keys().to_numpy()[1:].copy()
    
    dataHere_Imputed_DICT = {'subjectkey':dataHere_ToImpute_Final_TrainTest['subjectkey'].to_numpy()}
    for keyIx in range(keysHere_All.shape[0]):
        keyStr = keysHere_All[keyIx]
        allOtherKeys = keysHere_All.copy()
        allOtherKeys = np.delete(allOtherKeys,keyIx)

        yData_Orig = pd.DataFrame({keyStr:dataHere_ToImpute_Final_TrainTest[keyStr].to_numpy()})[keyStr].copy()
        nanIxs = np.where(np.isnan(yData_Orig))[0]

        xData = dataHere_ToImpute_Final_TrainTest[allOtherKeys].values.copy()

        if nanIxs.shape[0]!=0:
            #print(f"Predicting {keyStr}...")

            if keyStr in keysHere_Trans2_Adj: # Categorical (binarized) 
                yData = mode_imputation(yData_Orig).to_numpy().copy() # for categorical  data

                xData_Train = xData[subjsTrain,:].copy()
                xData_Test = xData[subjsTest,:].copy()

                yData_Train = yData[subjsTrain].copy()
                yData_Test = yData[subjsTest].copy()

                modelHere = RandomForestClassifier()
                modelHere.fit(xData,yData)

            elif keyStr not in keysHere_Trans2_Adj: # Continuous 
                #yData = mean_median_imputation(yData_Orig,kind='mean').to_numpy().copy() # for continuous data
                yData = mean_median_imputation(yData_Orig,kind='median').to_numpy().copy() # for continuous data

                xData_Train = xData[subjsTrain,:].copy()
                xData_Test = xData[subjsTest,:].copy()

                yData_Train = yData[subjsTrain].copy()
                yData_Test = yData[subjsTest].copy()

                modelHere = RandomForestRegressor()
                modelHere.fit(xData,yData)

            yPred = modelHere.predict(xData_Test[:nanIxs.shape[0],:]) # values 
            yData_Imputed = yData_Orig.copy()
            yData_Imputed[nanIxs] = yPred

        else: 
            print(f"{keyStr} is not missing data, skipping...")
            yData_Imputed = yData_Orig.copy()

        dataHere_Imputed_DICT[keyStr] = yData_Imputed.copy()

    dataHere_Imputed = pd.DataFrame(dataHere_Imputed_DICT)

    nanIxsDF = dataHere_Imputed.isnull().sum()[dataHere_Imputed.isnull().sum() != 0].index # use ==0 to get not-nans
    if nanIxsDF.shape[0] != 0: 
        print(f"There is still missing data points, please check.")
        
    if saveResults:
        if reNormData:
            if useZ:
                dataHere_Imputed.to_csv(saveDir + 'dataHere_Normed_Z_Transformed_Imputed.csv')
            elif not useZ:
                dataHere_Imputed.to_csv(saveDir + 'dataHere_Normed_MinMax_Transformed_Imputed.csv')
        elif not reNormData:
            dataHere_Imputed.to_csv(saveDir + 'dataHere_NotNormed_Imputed.csv')

    return dataHere_Imputed

####################################################################################
def fingerprints_distance_Cocuzza(dataHere_Imputed,startIx_CSV=2):
    '''
    The goal of this step is to estimate individual differences correlation. Here, each variable-variable pair (i.e., each pair of behavioral measures) is correlated across subjects. So the resulting correlation score indicates the extent that individual differences (across those 2 given measures) are similar. NOTE: we used spearman rho because of it's inference for rank correlation that doesn't assume specific distributions a priori (i.e., is less parametric than r)
    
    INPUTS:
        dataHere_Imputed: output dataframe from prior step/function (imputation step) above.
        startIx_CSV: an artifact of pandas is that sometimes an indexing column is added. You always want this to be at least 1 (because we want to skip the true 1st colum which is subject IDs), but maybe increase to 2 (default) or even 3 etc. if more columns were added to left of that subjectkey column.
        
    OUTPUTS: 
        behavCorrs_IndivDiffs: an adjacency matrix (square, symmetric). x/y dimensions correspond to behavioral variables in input dataframe. 
    '''

    # Correlation matrix: individual differences across subjects 
    numVarsCND = dataHere_Imputed.shape[1]-startIx_CSV
    behavCorrs_IndivDiffs = np.zeros((numVarsCND,numVarsCND))
    for metricIx in range(numVarsCND):
        thisTargetKey = dataHere_Imputed.keys().to_numpy()[startIx_CSV:][metricIx]

        for metricIx_Next in range(numVarsCND):
            thisSourceKey = dataHere_Imputed.keys().to_numpy()[startIx_CSV:][metricIx_Next]

            r,p = stats.spearmanr(dataHere_Imputed[thisTargetKey].to_numpy(),dataHere_Imputed[thisSourceKey].to_numpy(),nan_policy='omit')
            behavCorrs_IndivDiffs[metricIx,metricIx_Next] = r

    np.fill_diagonal(behavCorrs_IndivDiffs,0)
    
    return behavCorrs_IndivDiffs

####################################################################################
def fingerprints_clustering_Cocuzza(behavCorrs_IndivDiffs,dataHere_Imputed,nClusters=4,startIx_CSV=2):
    '''
    The goal of this step is to perform heirarchical agglomerative clustering on individual differences correlations. See cnd_project_clustering_checks_pipe.Rmd for example usage of. Rpackage that helps optimize number of clusters to use.
    
    INPUTS:
        behavCorrs_IndivDiffs: output matrix from step/function above (distance step).
        dataHere_Imputed: output dataframe from step/function above (imputation step).
        nClusters: Number of clusters to use; Default = 4; based on R package nbClust. 
        startIx_CSV: an artifact of pandas is that sometimes an indexing column is added. You always want this to be at least 1 (because we want to skip the true 1st colum which is subject IDs), but maybe increase to 2 (default) or even 3 etc. if more columns were added to left of that subjectkey column.

    OUTPUTS: 
        clusterBoundaries: array of size nClusters x 3: column 1 = start ix, column 2 = stop ix (py --> +1), column 3 = number of features in cluster.
        clusterOrder: variable that assigns each variable from the input dataframe to a cluster. 
    '''
    
    originalOrder = dataHere_Imputed.keys().to_numpy()[startIx_CSV:].copy()
    scoreNames = dataHere_Imputed.keys().to_numpy()[startIx_CSV:].copy()
    
    linkagMatrix = scipy.cluster.hierarchy.ward(behavCorrs_IndivDiffs) # same as colLink
    cutTree = scipy.cluster.hierarchy.cut_tree(linkagMatrix, n_clusters = nClusters).flatten() # note: n_clusters can be 2D, will return 2D group membership 
    percentileList = pd.DataFrame({'column_names':scoreNames,'cluster_membership':cutTree})
    clusterList = percentileList.sort_values(by='cluster_membership')

    clusterBoundaries = np.zeros((nClusters,3)) # Col 1 = start ix, Col 2 = stop ix (py --> +1), Col 3 = number of features in cluster 
    clusterOrder = np.zeros((originalOrder.shape[0])) # Use this as a sorting/indexing variable 
    for clusterIx in range(nClusters):
        if not useHandOrder:
            clusterSetIxs = np.where(clusterList['cluster_membership'].to_numpy() == clusterIx)[0]
        elif useHandOrder:
            clusterSetIxs = np.where(clusterListAdj['cluster_membership'].to_numpy() == clusterIx)[0] # NOTE: hand-ordered above 
        clusterBoundaries[clusterIx,0] = clusterSetIxs[0]
        clusterBoundaries[clusterIx,1] = clusterSetIxs[-1] + 1
        clusterBoundaries[clusterIx,2] = clusterSetIxs.shape[0]

        if not useHandOrder:
            variablesThisCluster = clusterList['column_names'].to_numpy().copy()
        elif useHandOrder:
            variablesThisCluster = clusterListAdj['column_names'].to_numpy().copy() # NOTE: hand-ordered above

        print(f"\nCluster {clusterIx+1}:\n{clusterList[clusterList['cluster_membership']==clusterIx]['column_names'].to_numpy()}")

        for varIx in range(variablesThisCluster.shape[0]):
            thisVar = variablesThisCluster[varIx]
            origIx = np.where(originalOrder==thisVar)[0]
            if not useHandOrder:
                newIx = np.where(clusterList['column_names'].to_numpy()==thisVar)[0]
            elif useHandOrder:
                newIx = np.where(clusterListAdj['column_names'].to_numpy()==thisVar)[0] # NOTE: hand-ordered above
            #clusterOrder[varIx] = origIx
            clusterOrder[newIx] = origIx

    # NOTE: can verify with: 
    # np.array_equal(originalOrder[clusterOrder.astype(int)],clusterList['column_names'].to_numpy())

    # NOTE: an alternative with sklearn; results are effectively equivalent to scipy method above:
    #numClustersStr = 4
    # Appending distance between clusters at every iteration in for loop
    #hClustering_Distances_All = []
    #hClustering_CH_Index_All = []
    #for numClusters in range(2,numVarsCND):
    #    hClustering = sklearn.cluster.AgglomerativeClustering(n_clusters=numClusters, metric='euclidean', linkage='ward', compute_distances=True).fit(behavCorrs_IndivDiffs)
    #    hClustering_Distances_All.append(hClustering.distances_)
    #    hClustering_CH_Index_All.append(sklearn.metrics.calinski_harabasz_score(behavCorrs_IndivDiffs,hClustering.labels_))
    #hClustering_CH_Index_All = np.asarray(hClustering_CH_Index_All)
    #hClustering_Distances = hClustering_Distances_All[0].copy() # note: all sub-arrays appear to be equal? why do loop? this appears to be equal to colLink
    #xNumClusters = np.array([indexHere for indexHere in range(numVarsCND,1,-1)]) # number of clusters in descending oder 
    
    return clusterBoundaries, clusterOrder

####################################################################################
def fingerprints_PCA_Cocuzza(behavCorrs_IndivDiffs,dataHere_Imputed,numClusters=4, startIx_CSV=2,printTestName=True,numComps=1,useAbs=Fale):
    '''
    After clustering has been performed (see above), run PCA on the original data (i.e.., not the distance matrix), one cluster at a time. In the manuscript we focused on the 1st PC for each cluster.
    
    INPUTS:
        behavCorrs_IndivDiffs: output matrix from step/function above (distance step).
        dataHere_Imputed: output dataframe from step/function above (imputation step).
        numClusters: Number of clusters to use; Default = 4; based on R package nbClust. 
        startIx_CSV: an artifact of pandas is that sometimes an indexing column is added. You always want this to be at least 1 (because we want to skip the true 1st colum which is subject IDs), but maybe increase to 2 (default) or even 3 etc. if more columns were added to left of that subjectkey column.
        printTestName: Default True; whether to include test names in the print 
        numComps: Default 1; For PCA, either use a number (less than # features) or None
        useAbs: Default False; Use absolute values of PCA scores for visualizing 


    OUTPUTS: 
        pcaScores_All: array of size number of subjects x number of clusters. This gives log-likelihood of each subject to express each cluster. 
        loadings_All: dictionary where keys = cluster IDs; in each, an array of factor loadings corresponding to measures in that cluster  
        
    '''    
    
    modelHere = sklearn.cluster.AgglomerativeClustering(n_clusters=numClusters, metric='euclidean', linkage='ward', compute_distances=True)
    modelHere.fit_predict(behavCorrs_IndivDiffs)
    labelsHere = modelHere.labels_

    pcaScores_All = np.zeros((dataHere_Imputed.shape[0],numClusters))
    loadings_All = {} # keys given by cluster, e.g., 'cluster_1', etc. 
    for labelIx in range(numClusters):
        tempList = []
        clusterIxsHere = np.where(labelsHere==labelIx)[0]
        testNamesHere = dataHere_Imputed.keys().to_numpy()[startIx_CSV:][clusterIxsHere]
        dataHere_Cluster = dataHere_Imputed[testNamesHere].copy()
        if numComps is not None:
            modelHere = sklearn.decomposition.PCA(n_components=numComps)
        else:
            modelHere = sklearn.decomposition.PCA()
        modelHere.fit(dataHere_Cluster)
        modelHere_Transform = modelHere.fit_transform(dataHere_Cluster).copy() # samples (subjects) x components 

        # NOTE: other variables are here that aren't returned; feel free to edit to save out/return for other research studies 
        compsHere = modelHere.components_.copy() # components x features (psych measures)
        explVarHere = modelHere.explained_variance_.copy() # components 
        explVarRatioHere = modelHere.explained_variance_ratio_.copy() # components 
        inverseTransformHere = modelHere.inverse_transform(modelHere_Transform).copy() # samples (subjects) x features (psych measures) -> original space 
        pcaScoreHere = modelHere.score_samples(dataHere_Cluster).copy() # samples (subjects) -> NOTE: this is the log-likelihood of each sample (subject)
        loadingsHere = compsHere.T * np.sqrt(explVarHere) # features (psych measures) x components 

        print(f"Cluster {labelIx+1}: the 1st PC explains {np.round(explVarRatioHere[0]*100,2)}% of the variance (of {np.round(np.nansum(explVarRatioHere)*100,2)}% total).")

        pcaScores_All[:,labelIx] = pcaScoreHere.copy()
        loadings_All['cluster_'+str(labelIx+1)] = loadingsHere.copy()
        
    return pcaScores_All, loadings_All
    
####################################################################################
# Helper function to perform min-max normalization. Also known as min-max scaling and feature scaling 
def minMaxNorm(data,minVal=0,maxVal=1):
    normedData = minVal + (((data-np.nanmin(data))*(maxVal-minVal))/(np.nanmax(data)-np.nanmin(data)))
    return normedData

####################################################################################
# Helper function for imputation
def add_label(data,attr,name_notnan = 'Training', name_nan = 'Predict'):
    null_pos = data[attr][data[attr].isnull()].index
    data['Label'] = name_notnan
    data['Label'].iloc[null_pos] = name_nan
    
####################################################################################
# Helper function for imputation
def mean_median_imputation(data, kind = 'mean'):
    if kind=='mean':
        return data.fillna(np.mean(data.dropna()))
    elif kind == 'median': 
        return data.fillna(np.median(data.dropna()))

####################################################################################
# Helper function for imputation 
def mode_imputation(data):
    return data.fillna(stat.mode(data))