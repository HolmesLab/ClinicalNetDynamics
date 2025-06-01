# C.V. Cocuzza, 2025. Data splitting code for "Brain network dynamics reflect psychiatric illness status and transdiagnostic symptom profiles across health and disease". Cocuzza et al., 2025. 

# Split data into train/test/validation sets to prevent data leakage. 
# See here: Rosenblatt, M., Tejavibulya, L., Jiang, R., Noble, S., & Scheinost, D. (2024). Data leakage inflates prediction performance in connectome-based machine learning models. Nature Communications, 15(1), 1829. https://doi.org/10.1038/s41467-024-46150-w

# Note: this isn't a function, just example code that can be adapted for other data (given that this is an important consideration for predictive modeling) 

# Note: this required extra considerations: accounting for collection site and primary diagnosis groups (i.e., evenly allocate as best as possible)
# Can extend the ideas here to account for: age, sex, SES, etc. 

# Note: this script is rather long because I included lots of QC 

########################################################################################
# Imports (some may need to be installed) 
import numpy as np
import os
import sys
import pandas as pd

########################################################################################
# Variables that can/should be changed for different data and project aims

trainTestPercent = 80 
trainPercent = 80 # i.e., this is 80% of the original 80% above

dirHere = '/directory/path/to/save/indexing/vectors/' # changed from RU HPC paths for privacy 

subjIxsToUse = subjIxs_Adj_States.copy() # This is a final vector of subject indices, adjusted for the fMRI runs/states of interest
subjIDsToUse = subjIDs_All_NDA_Adj.copy() # This is a final vector of subject IDs (strings) (corresponds to subjIxsToUse) 

dataHere_ToImpute_Final_CND = dataHere_ToImpute_Final.copy() # This is a pre-loaded dataframe with subject IDs for QX 

demoData_Init = pd.read_csv(dirHere + 'dataset_info_tcp_cnd_nda.csv').copy() # This is another dataframe; has demographic info for study

# Keys in the demoData_Init dataframe 
subjVarStr = 'subjectkey' # has subject IDs 
caseContVarStr = 'case_control' # whether patient or healthy control 
primDXVarStr = 'primary_dx' # primary diagnosis if applicable
secDXVarStr = 'secondary_dx' # secondary diagnosis (n/a for some participants) 
raceVarStr = 'race' 
ethnVarStr = 'ethnicity'
sexVarStr = 'sex' 
ageVarStr = 'age_years'

########################################################################################
# Set train/test/validation percents; done across subjects in this study 

# First divide into validation (held-out) and train/test (1 chunk for now) 
validationPercent = int(100 - trainTestPercent)

# Then set train/test percents for subsampling during supervised learning steps 
testPercent = int(100 - trainPercent) 

########################################################################################
# Then over subject indices (which are likely to be fixed in a certain order, based on subject ID), 
# generate 1 vector of held-out validation set indices and many random folds of train/test subsamples 

########################################################################################
# First adjust for states (fMRI runs) of-interest (based on subjIxsToUse and subjIDsToUse)
# And use dataHere_ToImpute_Final_CND dataframe for QC (making sure the subjIDsToUse vector is accurate) 

subjIDs_DF = dataHere_ToImpute_Final_CND['subjectkey'].to_numpy().copy()

subjIDs_NonCND = [] # log the subject IDs that we'll remove (i.e., don't have fMRI data of interest) 
for subjIx in range(subjIDs_DF.shape[0]):
    subjIDHere_DF = subjIDs_DF[subjIx]
    
    if subjIDHere_DF not in subjIDsToUse:
        subjIDs_NonCND.append(subjIDHere_DF)
subjIDs_NonCND = np.asarray(subjIDs_NonCND)

for subjIx in range(subjIDs_NonCND.shape[0]): # adjust dataframe to the final set of subjects 
    thisID_RM = subjIDs_NonCND[subjIx]
    dataHere_ToImpute_Final_CND = dataHere_ToImpute_Final_CND.drop(dataHere_ToImpute_Final_CND[dataHere_ToImpute_Final_CND['subjectkey']==thisID_RM].index)

# Set numbers of subjects in each train/test/validation and print 
nSubjsTrainTest = np.round((dataHere_ToImpute_Final_CND.shape[0]*trainTestPercent)/100)
nSubjsValid = np.round((dataHere_ToImpute_Final_CND.shape[0]*(100-trainTestPercent))/100)
nSubjsTrain = np.round((nSubjsTrainTest * trainPercent)/100)
nSubjsTest = nSubjsTrainTest - nSubjsTrain

print(f"Final dataset for imputation: {dataHere_ToImpute_Final_CND.shape[0]} subjects by {dataHere_ToImpute_Final_CND.shape[1]} measures. "+
      f"{trainTestPercent}% subjects (n = {nSubjsTrainTest}) will be used for train/test and "+
      f"{100-validationPercent}% subjects (n = {nSubjsValid}) held out for validation (final analyses).")

########################################################################################
# Now take out semi-random 20% for validation later 
# Non-random parts: make sure proportions of testing site, case-control group, and DX (for cases) are similar across train/test and validation
# (to the best of our ability) 

subjVec = []
caseContVec = []
primDXVec = []
secDXVec = []
raceVec = []
ethnVec = []
sexVec = []
ageVec = []

for subjIx in range(subjIDsToUse.shape[0]):
    subjIDHere = subjIDsToUse[subjIx]
    subjIxHere = subjIxsToUse[subjIx] 
    
    demoData_ThisSubj = demoData_Init[demoData_Init[subjVarStr]==subjIDHere].copy()
    
    # Fill in empty lists from above with variables of interest 
    subjVec.append(demoData_ThisSubj[subjVarStr].to_numpy()[0]) # NOTE: IDs have "_PCM" and "_PCY" for McLean and Yale sites, respectively - will use for site allocation below
    caseContVec.append(demoData_ThisSubj[caseContVarStr].to_numpy()[0])
    primDXVec.append(demoData_ThisSubj[primDXVarStr].to_numpy()[0])
    secDXVec.append(demoData_ThisSubj[secDXVarStr].to_numpy()[0])
    raceVec.append(demoData_ThisSubj[raceVarStr].to_numpy()[0])
    ethnVec.append(demoData_ThisSubj[ethnVarStr].to_numpy()[0])
    sexVec.append(demoData_ThisSubj[sexVarStr].to_numpy()[0])
    ageVec.append(demoData_ThisSubj[ageVarStr].to_numpy()[0])
        
demoData_Dict = {subjVarStr:subjVec,
                 caseContVarStr:caseContVec,
                 primDXVarStr:primDXVec,
                 secDXVarStr:secDXVec,
                 raceVarStr:raceVec,
                 ethnVarStr:ethnVec,
                 sexVarStr:sexVec,
                 ageVarStr:ageVec}

demoData = pd.DataFrame(demoData_Dict)

####################################################################################
# Find approximate percentages of each variable that we want to make sure to divide evenly (as best as possible)

####################################################################################
# COLLECTION SITE (McLean/Yale)

siteCount_Yale = 0
siteCount_McLean = 0 
siteVec_All = []

for subjIx in range(len(subjVec)):
    if 'PCY' in subjVec[subjIx]:
        siteCount_Yale = siteCount_Yale + 1 
        siteVec_All.append('Yale')
        
    elif 'PCM' in subjVec[subjIx]:
        siteCount_McLean = siteCount_McLean + 1 
        siteVec_All.append('McLean')
        
    else:
        print(f"Subject {subjIx} (py ix) ({subjVec[subjIx]}) not identified as Yale or McLean, please check.")
        siteVec_All.append('NONE_ERROR')
        
perc_Site_Yale = np.round((siteCount_Yale*100)/len(subjVec),2)
perc_Site_McLean = np.round((siteCount_McLean*100)/len(subjVec),2)
print(f"\nOf {len(subjVec)} subjects, n = {siteCount_Yale} ({perc_Site_Yale}%) from Yale and n = {siteCount_McLean} ({perc_Site_McLean}%) from McLean.")

####################################################################################
# DIAGNOSTIC STATUS (CASE/CONTROL) 

groupCount_Control = 0
groupCount_Patient = 0 

for caseContLabel in range(len(caseContVec)):
    if 'Control' in caseContVec[caseContLabel]:
        groupCount_Control = groupCount_Control + 1 
        
    elif 'Patient' in caseContVec[caseContLabel]:
        groupCount_Patient = groupCount_Patient + 1 
        
    else:
        print(f"Subject {caseContLabel} (py ix) ({caseContVec[caseContLabel]}) not identified as Control or Patient, please check.")
        
perc_group_Control = np.round((groupCount_Control*100)/len(caseContVec),2)
perc_group_Patient = np.round((groupCount_Patient*100)/len(caseContVec),2)
print(f"\nOf {len(caseContVec)} subjects, n = {groupCount_Control} ({perc_group_Control}%) are healthy controls and n = {groupCount_Patient} ({perc_group_Patient}%) are patients.\n")

####################################################################################
# PRIMARY DIAGNOSIS (only applies to Patient group from above) 

primDXVec_Patients = [] # First pull out only patients 
primDXVec_All = [] # here, 'none' will = HCs

for subjIx in range(len(caseContVec)):
    subjIDHere = subjVec[subjIx]
    
    if 'Patient' in caseContVec[subjIx]:
        primDXHere = demoData[primDXVarStr].to_numpy()[subjIx]
        if type(primDXHere)!=str:
            if np.isnan(primDXHere):
                print(f"***** subject {subjIx} (py ix) ({subjIDHere}) labeled as {caseContVec[subjIx]}, but primary diagnosis is listed as {primDXHere}, please check.")
                primDXHere = 'general_notsorted'
                
        primDXVec_Patients.append(primDXHere)
        primDXVec_All.append(primDXHere)
        
    elif 'Control' in caseContVec[subjIx]:
        primDXVec_All.append('none')
        
    else:
        print(f"***** subject {subjIx} (py ix) ({subjIDHere}) not labeled as patient or control, please check.")
        
primDXVec_Patients = np.asarray(primDXVec_Patients).copy()
primDXVec_All = np.asarray(primDXVec_All).copy()
# NOTE: to look at labels, see: np.unique(primDXVec_Patients)

####################################################################################
# Now group together; note DXs are intentionally chunked over more general categories (see in-loop notes below)
# e.g., CUD (cannabis use disorder) and AUD (alcohol use disorder) both in general SUD (substance use disorder) 
# This is described in the manuscript and was discussed/agreed upon with clinician coauthors 

# BP = bipolar (1 or 2 OR cyclothymia); SCZ_SZA = schizophrenia OR schizoaffective; 
# ED = eating disorder (any kind); SUD = substance use disorder (any kind); 'PD' = panic disorder (****** including phobias)
# other_rare to include: PMDD (premenstrual dysphoric disorder), not labeled DXs ("general_notsorted"), 
# none: healthy control

dxLabels = np.asarray(['ADHD', 'BP', 'Depression', 'OCD', 'Anxiety', 'PTSD', 'SCZ_SZA','ED', 'SUD', 'PD','other_rare','none'])
dxLabelCounts = np.zeros((dxLabels.shape[0])) # counts in each index corresponds to above label at same index 
primDXVec_All_Adj = []

for subjIx in range(primDXVec_All.shape[0]):
    thisPrimDX = primDXVec_All[subjIx]
    
    if 'ADHD' in thisPrimDX: 
        # 'ADHD', 'mild ADHD'
        labelHere = 'ADHD'
        
    elif 'BP' in thisPrimDX or 'cyclothymia' in thisPrimDX: 
        # 'BP1', 'BP2', 'BPI', 'BPII', 'cyclothymia'
        labelHere = 'BP'
        
    elif 'MDD' in thisPrimDX or 'ysthymia' in thisPrimDX or 'depression' in thisPrimDX: 
        # 'Dysthymia', 'Past dysthymia', 'dysthymia', 'MDD', 'MDD (w/ mild anxious distress, melancholic features)', 'Past MDD', 'past MDD', 'past MDD (due to SUD)', 'depression NOS'
        labelHere = 'Depression'
        if 'PMDD' in thisPrimDX:
            labelHere = 'other_rare'
        
    elif 'OCD' in thisPrimDX: 
        # 'OCD', 'Past OCD'
        labelHere = 'OCD'

    elif 'GAD' in thisPrimDX or 'anxiety' in thisPrimDX: 
        # 'GAD', 'past GAD', 'anxiety NOS', 'Social anxiety', 'social anxiety'
        labelHere = 'Anxiety'
        
    elif 'PTSD' in thisPrimDX: 
        # 'PTSD', 'past PTSD',
        labelHere = 'PTSD'
        
    elif 'SZ' in thisPrimDX: 
        # 'SZ','SZ ', 'SZ (ruleout SZA)', 'SZ or SZA', 'SZA',
        labelHere = 'SCZ_SZA'
        
    elif 'ED' in thisPrimDX or 'eating' in thisPrimDX: 
        # 'past unspec ED', 'binge eating disorder', 
        if 'general_notsorted' not in thisPrimDX:
            labelHere = 'ED'
        
    elif 'UD' in thisPrimDX or 'polysub' in thisPrimDX: 
        # 'mild CUD', 'moderate AUD', 'moderate CUD', 'past mild AUD', 'past moderate AUD', 'past severe CUD', 'present polysub use (ER)', 'severe AUD'
        if 'past MDD (due to SUD)' not in thisPrimDX:
            labelHere = 'SUD'
            
    elif 'panic' in thisPrimDX or 'phobia' in thisPrimDX: 
        # 'Past agoraphobia with panic disorder', 'panic disorder', 'specific phobia'
        labelHere = 'PD'

    elif 'general_notsorted' in thisPrimDX:
        labelHere = 'other_rare'

    elif 'none' in thisPrimDX:
        labelHere = 'none'
        
    else:
        print(f" ***** {thisPrimDX} not sorted, please check.")

    ixHere = np.where(dxLabels==labelHere)[0]
    dxLabelCounts[ixHere] = dxLabelCounts[ixHere] + 1 
    primDXVec_All_Adj.append(labelHere)

#if np.nansum(dxLabelCounts) != primDXVec_Patients.shape[0]: print(f"***** primary diagnosis label counts off, please check.")

perc_prim_DX = np.round((dxLabelCounts*100)/primDXVec_Patients.shape[0],2)
print('')
for primDXIx in range(dxLabels.shape[0]):
    labelHere = dxLabels[primDXIx]
    if labelHere != 'none':
        countHere = dxLabelCounts[primDXIx]
        percHere = perc_prim_DX[primDXIx]
        print(f"{labelHere}: n = {int(countHere)} or {percHere}% of all patients (total patients, n = {primDXVec_Patients.shape[0]}).")

demoData_Final = demoData.copy()
demoData_Final.insert(demoData_Final.shape[1],'primary_dx_adjusted',primDXVec_All_Adj,True)
demoData_Final.insert(demoData_Final.shape[1],'site',siteVec_All,True)

####################################################################################
# NOW ALLOCATE TO VALIDATION or TRAIN/TEST and SAVE 
# NOTE: train/test subsampled for k-fold CV in later analyses 
print("\n***************************************************************************\n")

subjKeys = demoData_Final['subjectkey'].to_numpy().copy()
dxKeys = demoData_Final['primary_dx_adjusted'].to_numpy().copy()
siteKeys = demoData_Final['site'].to_numpy().copy()
subjIxs_Control = np.where(dxKeys=='none')[0]
subjIxs_Patient = np.where(dxKeys!='none')[0]
# NOTE: subjIxs_Patient.shape[0] + subjIxs_Control.shape[0] should = total number of subjects in sample used 

subjKeys_Control = subjKeys[subjIxs_Control].copy()
subjKeys_Patient = subjKeys[subjIxs_Patient].copy()

# First deal with controls: maintain site ratio in 80/20 split 
siteKeys_Control = siteKeys[subjIxs_Control].copy()
siteIxs_Yale = np.where(siteKeys_Control=='Yale')[0]
siteIxs_McLean = np.where(siteKeys_Control=='McLean')[0]
np.random.shuffle(siteIxs_McLean)
np.random.shuffle(siteIxs_Yale)
trainNum_McLean = np.round((siteIxs_McLean.shape[0] * trainTestPercent) / 100)
trainNum_Yale = np.round((siteIxs_Yale.shape[0] * trainTestPercent) / 100)
trainIxs_McLean = siteIxs_McLean[:int(trainNum_McLean)].copy()
trainIxs_Yale = siteIxs_Yale[:int(trainNum_Yale)].copy()
validIxs_McLean = siteIxs_McLean[int(trainNum_McLean):].copy()
validIxs_Yale = siteIxs_Yale[int(trainNum_Yale):].copy()

# Initialize 
subjKeys_TrainTest = np.concatenate((subjKeys_Control[trainIxs_Yale],subjKeys_Control[trainIxs_McLean]))
subjKeys_Valid = np.concatenate((subjKeys_Control[validIxs_Yale],subjKeys_Control[validIxs_McLean]))

# Second deal with patients: maintain 80/20 split in each primary dx group 
dxKeys_Patient = dxKeys[subjIxs_Patient].copy()
subjKeys_RandAlloc = []
for dxLabelIx in range(dxLabels.shape[0]):
    thisDxLabel = dxLabels[dxLabelIx]
    if thisDxLabel != 'none':
        dxLabelIxs = np.where(dxKeys_Patient==thisDxLabel)[0]
        subjKeysHere = subjKeys_Patient[dxLabelIxs]
        
        if dxLabelIxs.shape[0] < 5: # NOTE: 5 is the smallest number to give 80/20 split, so will collect these and randomly allocate later 
            if len(subjKeys_RandAlloc) == 0:
                subjKeys_RandAlloc = subjKeysHere.copy()
            else: 
                subjKeys_RandAlloc = np.concatenate((subjKeys_RandAlloc,subjKeysHere))

        elif dxLabelIxs.shape[0] >= 5:
            np.random.shuffle(dxLabelIxs)
            numTrainHere = np.round((dxLabelIxs.shape[0] * trainTestPercent) / 100)
            trainIxsHere = dxLabelIxs[:int(numTrainHere)].copy()
            validIxsHere = dxLabelIxs[int(numTrainHere):].copy()
            subjKeysHere_Train = subjKeys_Patient[trainIxsHere].copy()
            subjKeysHere_Valid = subjKeys_Patient[validIxsHere].copy()

            subjKeys_TrainTest = np.concatenate((subjKeys_TrainTest,subjKeysHere_Train))
            subjKeys_Valid = np.concatenate((subjKeys_Valid,subjKeysHere_Valid))

# Now add back in dx groups with n<5 
randAllocNum = subjKeys_RandAlloc.shape[0]
randAllocIxVec = np.arange(randAllocNum)
np.random.shuffle(randAllocIxVec)
numTrainRandAlloc = np.round((randAllocIxVec.shape[0] * trainTestPercent) / 100)
trainIxsRandAlloc = randAllocIxVec[:int(numTrainRandAlloc)].copy()
validIxsRandAlloc = randAllocIxVec[int(numTrainRandAlloc):].copy()
subjKeys_TrainRandAlloc = subjKeys_RandAlloc[trainIxsRandAlloc].copy()
subjKeys_ValidRandAlloc = subjKeys_RandAlloc[validIxsRandAlloc].copy()

subjKeys_TrainTest = np.concatenate((subjKeys_TrainTest,subjKeys_TrainRandAlloc))
subjKeys_Valid = np.concatenate((subjKeys_Valid,subjKeys_ValidRandAlloc))

if subjKeys_TrainTest.shape[0] != np.unique(subjKeys_TrainTest).shape[0]:
    print(f"***** There are possibly duplicate subject IDs in train/test allotment, please check")
if subjKeys_Valid.shape[0] != np.unique(subjKeys_Valid).shape[0]:
    print(f"***** There are possibly duplicate subject IDs in validation allotment, please check")
if (subjKeys_Valid.shape[0] + subjKeys_TrainTest.shape[0]) != subjIxsToUse.shape[0]:
    print(f"***** Total of train/test/valid allotment does not equal total sample size, please check.")

demoData_Final_TrainTest = demoData_Final.copy()
demoData_Final_Valid = demoData_Final.copy()

for subjIx in range(subjKeys_TrainTest.shape[0]): # remove train/test set from valid set 
    subjIDHere = subjKeys_TrainTest[subjIx]
    demoData_Final_Valid = demoData_Final_Valid.drop(demoData_Final_Valid[demoData_Final_Valid['subjectkey']==subjIDHere].index)

for subjIx in range(subjKeys_Valid.shape[0]): # remove valid set from train/test set 
    subjIDHere = subjKeys_Valid[subjIx]
    demoData_Final_TrainTest = demoData_Final_TrainTest.drop(demoData_Final_TrainTest[demoData_Final_TrainTest['subjectkey']==subjIDHere].index)  

####################################################################################
# SAVE 
demoData_Final_TrainTest.to_csv(dirHere + 'dataset_info_tcp_cnd_nda_TrainTest_'+cndSampleVer+'.csv')
demoData_Final_Valid.to_csv(dirHere + 'dataset_info_tcp_cnd_nda_Valid_'+cndSampleVer+'.csv')

####################################################################################
# Print Summary of train/test variables:
numYale_TrainTest = np.where(demoData_Final_TrainTest['site'].to_numpy()=='Yale')[0].shape[0]
percYale_TrainTest = np.round((numYale_TrainTest * 100) / demoData_Final_TrainTest.shape[0],2)
numMcLean_TrainTest = np.where(demoData_Final_TrainTest['site'].to_numpy()=='McLean')[0].shape[0]
percMcLean_TrainTest = np.round((numMcLean_TrainTest * 100) / demoData_Final_TrainTest.shape[0],2)

print(f"TRAIN/TEST ALLOTMENT:")
print(f"\nOf {demoData_Final_TrainTest.shape[0]} subjects, n = {numYale_TrainTest} ({percYale_TrainTest}%) "+
      f"from Yale and n = {numMcLean_TrainTest} ({percMcLean_TrainTest}%) from McLean.")

numHC_TrainTest = np.where(demoData_Final_TrainTest['case_control'].to_numpy()=='Control')[0].shape[0]
percHC_TrainTest = np.round((numHC_TrainTest * 100) / demoData_Final_TrainTest.shape[0],2)
numDX_TrainTest = np.where(demoData_Final_TrainTest['case_control'].to_numpy()=='Patient')[0].shape[0]
percDX_TrainTest = np.round((numDX_TrainTest * 100) / demoData_Final_TrainTest.shape[0],2)

print(f"\nOf {demoData_Final_TrainTest.shape[0]} subjects, n = {numHC_TrainTest} ({percHC_TrainTest}%) "+
      f"are healthy controls and n = {numDX_TrainTest} ({percDX_TrainTest}%) are patients.")

print('')
for primDXIx in range(dxLabels.shape[0]):
    labelHere = dxLabels[primDXIx]
    if labelHere != 'none':
        countHere = np.where(demoData_Final_TrainTest['primary_dx_adjusted'].to_numpy()==labelHere)[0].shape[0]
        percHere = np.round((countHere * 100) / numDX_TrainTest,2)
        print(f"{labelHere}: n = {int(countHere)} or {percHere}% of all patients (total patients, n = {numDX_TrainTest}).")

print("\n***************************************************************************\n")

# Summary of validation variables:
numYale_Valid = np.where(demoData_Final_Valid['site'].to_numpy()=='Yale')[0].shape[0]
percYale_Valid = np.round((numYale_Valid * 100) / demoData_Final_Valid.shape[0],2)
numMcLean_Valid = np.where(demoData_Final_Valid['site'].to_numpy()=='McLean')[0].shape[0]
percMcLean_Valid = np.round((numMcLean_Valid * 100) / demoData_Final_Valid.shape[0],2)

print(f"VALIDATION ALLOTMENT:")
print(f"\nOf {demoData_Final_Valid.shape[0]} subjects, n = {numYale_Valid} ({percYale_Valid}%) "+
      f"from Yale and n = {numMcLean_Valid} ({percMcLean_Valid}%) from McLean.")

numHC_Valid = np.where(demoData_Final_Valid['case_control'].to_numpy()=='Control')[0].shape[0]
percHC_Valid = np.round((numHC_Valid * 100) / demoData_Final_Valid.shape[0],2)
numDX_Valid = np.where(demoData_Final_Valid['case_control'].to_numpy()=='Patient')[0].shape[0]
percDX_Valid = np.round((numDX_Valid * 100) / demoData_Final_Valid.shape[0],2)

print(f"\nOf {demoData_Final_Valid.shape[0]} subjects, n = {numHC_Valid} ({percHC_Valid}%) "+
      f"are healthy controls and n = {numDX_Valid} ({percDX_Valid}%) are patients.")

print('')
for primDXIx in range(dxLabels.shape[0]):
    labelHere = dxLabels[primDXIx]
    if labelHere != 'none':
        countHere = np.where(demoData_Final_Valid['primary_dx_adjusted'].to_numpy()==labelHere)[0].shape[0]
        percHere = np.round((countHere * 100) / numDX_Valid,2)
        print(f"{labelHere}: n = {int(countHere)} or {percHere}% of all patients (total patients, n = {numDX_Valid}).")
       