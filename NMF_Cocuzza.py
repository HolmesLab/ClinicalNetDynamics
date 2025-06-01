# C.V. Cocuzza, 2025. Non-negative matrix factorization (NMF) code for "Brain network dynamics reflect psychiatric illness status and transdiagnostic symptom profiles across health and disease". Cocuzza et al., 2025. 

# This code largely adapts the approach used in: 
# Khambhati, A. N., Medaglia, J. D., Karuza, E. A., Thompson-Schill, S. L., & Bassett, D. S. (2018). Subgraphs of functional brain networks identify dynamical constraints of cognitive control. PLoS Computational Biology, 14(7), e1006234. https://doi.org/10.1371/journal.pcbi.1006234

'''
# Example usage: FC data in variable fcAll: 434 regions x 434 regions x 3 runs or states x 100 subjects to be used in train/test
# And another 30 subjects in fcData_Validation (434 x 434 x 3 x 30)
# Can copy and paste into jupyter notebook, python script, etc. 

####################################################################################
import sys
import numpy as np 

# VARIABLES TO CHANGE 
nBoots = 2 # Recommended as outer loop 1, with 100-1000 resamples (use batch/parallel processing if possible)
alphaSweep = np.arange(0.2,0.5,0.2) # Recommended as outer loop 2; sweep more values in real use (use batch/parallel processing if possible)
betaSweep = np.arange(1.0,2.0,0.5) # Recommended as outer loop 3; sweep more values in real use (use batch/parallel processing if possible)
nRuns = 3 
dirScripts = '/my/directory/path/where/scripts/are/saved/'

# Import NMF code 
sys.path.insert(0, dirScripts)
import NMF_Cocuzza as nmfCode 

# Configure data for cross-validation for alpha and beta parameters 
percentTrain = 50
percentTest = int(100-percentTrain)

nSubjs = fcAll.shape[3] 
if nSubjs % 2 != 0: nSubjs = int(nSubjs-1) # odd numbers

reconstructionError_Test_All = np.zeros((alphaSweep.shape[0],betaSweep.shape[0],nBoots))
mantelAll_TestOrig_TestInv_All = np.zeros((nRuns,alphaSweep.shape[0],betaSweep.shape[0],nBoots))

for bootNum in range(nBoots):
    subjIxVec = np.arange(nSubjs)
    np.random.shuffle(subjIxVec)
    nTrain = int((nSubjs * percentTrain)/100)
    nTest = int((nSubjs * percentTest)/100)
    
    trainSubjs = subjIxVec[:nTrain].copy() 
    testSubjs = subjIxVec[nTrain:].copy() 
    
    fcData_Train = fcAll[:,:,:,trainSubjs].copy()
    fcData_Test = fcAll[:,:,:,testSubjs].copy()

    nmfInputArray_Train = nmfCode.NMF_format_data(fcData_Train)
    nmfInputArray_Test = nmfCode.NMF_format_data(fcData_Test)
    
    for alphaIx in range(alphaSweep.shape[0]):
        alphaHere_W = alphaSweep[alphaIx]
        for betaIx in range(betaSweep.shape[0]):
            betaLossHere = betaSweep[betaIx] 
            
            # Run cross-validation 
            _,reconstructionError_Test,_,_,_,_,mantelAll_TestOrig_TestInv,_ = nmfCode.NMF_cv(nmfInputArray_Train,
                                                                                               nmfInputArray_Test,
                                                                                               fcData_Train,
                                                                                               fcData_Test,
                                                                                               nRuns,
                                                                                               bootNum,
                                                                                               alphaHere_W,
                                                                                               betaLossHere)
            
            reconstructionError_Test_All[alphaIx,betaIx,bootNum] = reconstructionError_Test.copy()
            mantelAll_TestOrig_TestInv_All[:,alphaIx,betaIx,bootNum] = mantelAll_TestOrig_TestInv.copy()

# Print CV results 
r,c = np.where(np.nanmean(reconstructionError_Test_All,axis=2)==np.nanmin(np.nanmean(reconstructionError_Test_All,axis=2)))
print(f"Minimum reconstruction error (averaged over {nBoots} train/test folds) = "+
      f"{np.round(np.nanmin(np.nanmean(reconstructionError_Test_All,axis=2)),2)}. alpha = {alphaSweep[r]}, beta = {betaSweep[c]} ")

for runIx in range(nRuns):
    r,c = np.where(np.nanmean(mantelAll_TestOrig_TestInv_All[runIx,:,:,:],axis=2)==np.nanmax(np.nanmean(mantelAll_TestOrig_TestInv_All[runIx,:,:,:],axis=2)))
    print(f"Run/state {runIx}: max mantel r (averaged over {nBoots} train/test folds) = "+
          f"{np.round(np.nanmax(np.nanmean(mantelAll_TestOrig_TestInv_All[runIx,:,:,:],axis=2)),4)}. alpha = {alphaSweep[r]}, beta = {betaSweep[c]} ")
    
mantelAvg = np.nanmean(np.nanmean(mantelAll_TestOrig_TestInv_All,axis=3),axis=0)
r,c = np.where(mantelAvg==np.nanmax(mantelAvg))
print(f"Across-state average max mantel r (averaged over {nBoots} train/test folds) = "+
      f"{np.round(np.nanmax(mantelAvg),4)}. alpha = {alphaSweep[r]}, beta = {betaSweep[c]} ")


####################################################################################
# Test stability of k (number of subgraphs/coefficients) with identified alpha and beta from above 
# NOTE: you can sweep all 3 together but code needs to be adapted and it will take longer computationally

# Change these 
subgraphSweep = np.arange(2,20)
alphaHere_W = 0.2
betaLossHere = 1.0

reconstructionError_Test_k_All = np.zeros((subgraphSweep.shape[0],nBoots))
mantelAll_TestOrig_TestInv_k_All = np.zeros((nRuns,subgraphSweep.shape[0],nBoots))

for bootNum in range(nBoots):
    subjIxVec = np.arange(nSubjs)
    np.random.shuffle(subjIxVec)
    nTrain = int((nSubjs * percentTrain)/100)
    nTest = int((nSubjs * percentTest)/100)
    
    trainSubjs = subjIxVec[:nTrain].copy() 
    testSubjs = subjIxVec[nTrain:].copy() 
    
    fcData_Train = fcAll[:,:,:,trainSubjs].copy()
    fcData_Test = fcAll[:,:,:,testSubjs].copy()

    nmfInputArray_Train = nmfCode.NMF_format_data(fcData_Train)
    nmfInputArray_Test = nmfCode.NMF_format_data(fcData_Test)
    
    for subgraphIx in range(subgraphSweep.shape[0]):
        nSubgraphs = subgraphSweep[subgraphIx]
        reconstructionError_Test_k,mantelAll_TestOrig_TestInv_k = nmfCode.NMF_subgraph_stability(nmfInputArray_Train,
                                                                                               nmfInputArray_Test,
                                                                                               fcData_Train,
                                                                                               fcData_Test,
                                                                                               nRuns,
                                                                                               bootNum,
                                                                                               alphaHere_W,
                                                                                               betaLossHere,
                                                                                               nSubgraphs)
        
        reconstructionError_Test_k_All[subgraphIx,bootNum] = reconstructionError_Test_k.copy()
        mantelAll_TestOrig_TestInv_k_All[:,subgraphIx,bootNum] = mantelAll_TestOrig_TestInv_k.copy()

# Print CV results 
r = np.where(np.nanmean(reconstructionError_Test_k_All,axis=1)==np.nanmin(np.nanmean(reconstructionError_Test_k_All,axis=1)))[0]
print(f"Minimum reconstruction error (averaged over {nBoots} train/test folds) = "+
      f"{np.round(np.nanmin(np.nanmean(reconstructionError_Test_k_All,axis=1)),2)}. k = {subgraphSweep[r]} subgraphs.")

for runIx in range(nRuns):
    r = np.where(np.nanmean(mantelAll_TestOrig_TestInv_k_All[runIx,:,:],axis=1)==np.nanmax(np.nanmean(mantelAll_TestOrig_TestInv_k_All[runIx,:,:],axis=1)))[0]
    print(f"Run/state {runIx}: max mantel r (averaged over {nBoots} train/test folds) = "+
          f"{np.round(np.nanmax(np.nanmean(mantelAll_TestOrig_TestInv_k_All[runIx,:,:],axis=1)),4)}. k = {subgraphSweep[r]} subgraphs.")
    
mantelAvg = np.nanmean(np.nanmean(mantelAll_TestOrig_TestInv_k_All,axis=2),axis=0)
r = np.where(mantelAvg==np.nanmax(mantelAvg))[0]
print(f"Across-state average max mantel r (averaged over {nBoots} train/test folds) = {np.round(np.nanmax(mantelAvg),4)}. k = {subgraphSweep[r]} subgraphs.")

####################################################################################
# After tuning parameters, run NMF on held-out validation data 
alphaHere_W = 0.2
betaLossHere = 1.0
nSubgraphs = 7 
nRuns = 3 
nRegions = 434 

nmfInputArray_Validation = nmfCode.NMF_format_data(fcData_Validation)
weightedSubgraphs, subgraphExpressions, weightedSubgraphs_NetShape, FC_All_Reconstituted = nmfCode.NMF_validation(nmfInputArray_Validation,
                                                                                                                  nRuns,
                                                                                                                  nRegions,
                                                                                                                  alphaHere_W,
                                                                                                                  betaLossHere,
                                                                                                                  nSubgraphs)
'''

########################################################################################
# Imports (some may need to be installed) 
import numpy as np
import os
#import nibabel as nib
import sklearn 
from sklearn.decomposition._nmf import _beta_divergence
import mantel # https://github.com/jwcarr/mantel

########################################################################################
def NMF_format_data(fcData,
                    verbose=True):
    '''
    Function to format input arrays for NMF. In manuscript, this corresponds to: flattening and concatenating functional connectivity (FC) arrays across subjects and rest/task states. 

    IMPORTANT NOTE: this does not distinguish between train/test/validation sets. This should be done based on your dataset (and possibly looped outside of this function).

    Other notes:
    - We used FC arrays that were "pre-orderd" (i.e., sorting regions/parcels by functional network assignments) before extracting the upper triangle and vectorizing
    - This code will threshold for positive FC estimates (standard NMF use); negative-sensitive NMF variants exist, but were outside the scope of this project. 
    - This code assumes 2 dimensions to concatenate FC arrays over: subjects and states (i.e., fMRI runs). This can be adapted for other purposes. 
    - Here, we concatenate by state first, then order subjects within states. This can be adapted for other purposes.
    - In the manuscript, we used FC data only, but this can be adapted for use with multi-modal data (e.g., structural & functional connectivity, see Anderson et al. 2014 NeuroIm) 
    - Note that some authors have additionally ordered participants by other variables, such as age

    INPUTS: 
        fcData: connectivity matrices (weighted, symmetric); dimensions: regions x regions x states x subjects 
        verbose: If True, will print out some relevant info 

    OUTPUTS/RETURNS: 
        nmfInputArray: concatenated array to use as input into NMF; regional pairs x states & subjects
    '''
    
    nRegions = fcData.shape[0]
    nRuns = fcData.shape[2]
    nSubjs = fcData.shape[3]
    
    nRegionPairs = int(((nRegions * nRegions)/2) - (nRegions/2)) # how many pairs will be in upper triangle --> vectorized 

    initIxs = np.arange(0,(nRuns * nSubjs),nSubjs) # To keep track of concatenation indices
    nmfInputArray = np.zeros((nRegionPairs,(nRuns * nSubjs)))
    
    for subjIx in range(nSubjs):
        for runIx in range(nRuns): 

            fcHere = fcData[:,:,runIx,subjIx].copy()
            
            # Only positive values 
            fcHere[fcHere<0]=0

            # Use configuration matrix for this state 
            fcHere_Vector = fcHere[np.triu_indices(fcHere.shape[0],k=1)].reshape(-1,1).copy()

            if verbose and subjIx==0:
                print(f"For each subject: run {runIx}: upper triangle configuration matrix has {fcHere_Vector.shape[0]} features (brain connections)")        
                
            matAdjIx = initIxs[runIx]
            nmfInputArray[:,subjIx+matAdjIx] = fcHere_Vector[:,0].copy()

    if verbose:
        print(f"\nFull input matrix for NMF: {nmfInputArray.shape[0]} features "+
              f"(brain connections) and {nmfInputArray.shape[1]} samples ({nRuns} states or runs x {nSubjs} participants)")
        
    return nmfInputArray

########################################################################################
def NMF_cv(nmfInputArray_Train,
           nmfInputArray_Test,
           fcData_Train,
           fcData_Test,
           nRuns,
           bootNum,
           alphaHere_W,
           betaLossHere,
           
           nSubgraphs=7,
           
           l1RatioHere=0,
           initHere=None,
           solverHere='mu',
           tolHere=0.0001,
           maxIterHere=600,
           randomStateHere=None,
           alphaHere_H='same',
           verboseHere=0,
           shuffleHere=False):
    ''' 
    NMF cross-validation (CV) using train and test sets (i.e., randomly subsampling FC data into pre-defined percents -> running NMF_format_data() (see above) then calling this)

    NOTES: 
    - This assumes you have split your data into train/test (2 arrays) appropriately, possibly in an outer loop
    - In the paper, we did 50/50 train/test percentage splits over subjects (pseudo-randomized; we made sure ~evenly distributed diagnostic groups and collection sites) 
    - In the paper, we used 1000 bootstraps and used batch processing on Rutgers' high performance compute cluster
    - This CV is used to tune hyperparameters. To reduce overfitting, these parameters were used in all downstream analyses (on held-out validation set, on comparison models with different sinput states, etc.)

    - Also assumes outer loop(s) where beta and alpha (see below notes) are set iteratively; again we batched processing on RU HPC for speed
    - We tested alpha range of 0-2 (increments of 0.25); beta range of 0-2 (increments of 0.2); max beta = the Frobenius loss (constrained by sklearn)

    - The sklearn NMF function has a few "free" parameters; we'll be using CV for: 
    - (1) alpha W: multiplier for regularization terms for W matrix. Our use below uses same alpha for W and H (but "alpha H" could also be optimized; may overfit though).
    - (2) beta loss: the divergence to be minimized by measuring distance between X and dot product of WH. 

    - Note on beta by sklearn: they have presets at 1 ("kullback-leiber"), 2 ("frobenius"; default); and 0 ("itakura saito"). 
    - But, some initial tests show that any value (i.e., including decimals) between 0-2 can be used, but more granular values will make computation slower. 

    - Note: in Khambhati et al 2018 (PLOS Comp Biol); see Fig. S2. beta links with sparsity of subgraph edge weights, and alpha is regularization of temporal expression coefficients. 
    - In this paper, they did a similar CV (note they also swept # of components, or m, or # of subgraphs). 

    - Note: in Kalantar-Hormozi et al. 2023 (NeuroIm), they perform stability analysis for number of components (number of subgraphs, k). 
    - We'll do k stability tests separately from alpha/beta for computational feasibility, and run CV on the "middle ground" (based on literature 5-10) k=7, but this can be changed

    - Note: the L1 ratio will be set to 0 given some tests that other values (aside from very small, near zero values) lead to W/H solutions of all zeros. 
    - Per sklearn's docs, the L1 set to 0 actually is the L2 penalty, and L1 set to 1 is the L1 penalty, so the name is a bit misleading here, it's actually L2. 
    - L2 = ridge; shrinks parameters a bit but does not 0 them out. Lowers (but does not eradicate) the influence of less influential features. 
    - L1 = lasso; shrinks parameters toward 0. This may not work for this application b/c it's overly sparse. 
    - Note: sklearn describes the use of the regularizer here as L1 ratio: regularizaton a "mixing" parameter; 
    - between 0 to 1; if 0: L2 penalty (Frobenius norm), if 1: L2 penalty, if btwn 0-1: mix of L1 and L2.

    INPUTS:
        nmfInputArray_Train: output of NMF_format_data(), training set for this resampling. regional pairs x states & training subjects
        nmfInputArray_Test: output of NMF_format_data(), test set for this resampling. regional pairs x states & test subjects
        fcData_Train: input data for NMF_format_data(), training set for this resampling. regions x regions x runs or states x number of training subjects 
        fcData_Test: input data for NMF_format_data(), test set for this resampling. regions x regions x runs or states x number of test subjects 
        nRuns: number of runs/states in NMF input array (same as in NMF_format_data() and dimension 3 in fcData_Train and fcData_Test)
        bootNum: number for train/test resampling (set in outer loops)
        alphaHere_W: value of alpha for this parameter set/sweep (see notes above); regularization constant 
        betaLossHere: value of beta for this parameter set/sweep (see notes above); beta divergence to be minimized 
        nSubgraphs: set to a reasonable number (stabilized in later function) 
        l1RatioHere: initial tests show >0 overly sparsifies; so this is actually L2 ratio (when set to 0). Default 0
        initHere: default to None
        solverHere: may consider other options; note coordinate descent (cd) is sklearn default, but mu (multiplicative) is needed to sweep beta parameter. Default 'mu'
        tolHere: tolerance of stopping condition. Default 0.0001
        maxIterHere: iterations before timeout, 200 is sklearn default; I increased it based on warnings printed out --> Default 600
        randomStateHere: used only with init; or with coordinate descent. Default None
        alphaHere_H: Default 'same' --> use same alpha as sweeped by CV (for W matrix) for H matrix
        verboseHere: can set to True (1) if need to debug; Default 0 (False)
        shuffleHere: Used if you need to randomize coordiante order in solver; Default False 

    OUTPUTS/RETURNS:
        reconstructionError_Train: reconstruction error for training portion of model (reference if needed)
        reconstructionError_Test: reconstruction error (should be minimized) for trained model fit to test data *** recommended to measure accuracy 
        mantelAll_TrainOrig_TestOrig: mantel r similarity reference: train-test similarity, original FC arrays 
        mantelAll_TrainOrig_TrainInv: mantel r similarity reference: train-train similarity, original FC and NMF-FC
        mantelAll_TrainOrig_TestInv: mantel r similarity of train-original-FC and test-NMF-FC
        mantelAll_TestOrig_TrainInv: mantel r similarity of test-original-FC and train-NMF-FC
        mantelAll_TestOrig_TestInv: mantel r similarity of test-original-FC and test-NMF-FC *** recommended to measure accuracy 
        mantelAll_TrainInv_TestInv: mantel r similarity of train-NMF-FC and test-NMF-FC  *** possibly compare to mantelAll_TrainOrig_TestOrig

        * NOTE: each of the mantelAll_ output arrays above are r-like scores for runs or states; can also index p-value and z-score, but will need to set perms=100 or more
    '''
    
    nRegions = fcData_Train.shape[0] # should be equivalent for train and test 
    
    testDataHere = nmfInputArray_Train.copy()
    trainDataHere = nmfInputArray_Test.copy()
    
    nTestSubjs = int(testDataHere.shape[1] / nRuns)
    nTrainSubjs = int(trainDataHere.shape[1] / nRuns)
    nSubjsHere = nTestSubjs + nTrainSubjs
    
    # Perform NMF over sweeped parameter values and the current train/test split
    print(f"Hyperparameter pair here: alpha = {alphaHere_W}, beta = {betaLossHere}. Number of subgraphs = {nSubgraphs}.")

    # Implement NMF 
    nmfModel = sklearn.decomposition.NMF(n_components=nSubgraphs, 
                                         init=initHere, 
                                         solver=solverHere, 
                                         beta_loss=betaLossHere, 
                                         tol=tolHere, 
                                         max_iter=maxIterHere, 
                                         random_state=randomStateHere, 
                                         alpha_W=alphaHere_W, 
                                         alpha_H=alphaHere_H, 
                                         l1_ratio=l1RatioHere, 
                                         verbose=verboseHere, 
                                         shuffle=shuffleHere)

    nmfModel_FitTrain = nmfModel.fit(trainDataHere)
    weightedSubgraphs = nmfModel.fit_transform(trainDataHere)
    subgraphExpressions = nmfModel.components_
    
    # Reconstruction error (standard/built-in accuracy)
    reconstitutedArr = np.matmul(weightedSubgraphs,subgraphExpressions).copy()
    reconstructionError_Train_Auto = nmfModel.reconstruction_err_
    reconstructionError_Train = _beta_divergence(trainDataHere,weightedSubgraphs,subgraphExpressions,betaLossHere,square_root=True)
    
    if np.round(reconstructionError_Train_Auto,2)!=np.round(reconstructionError_Train,2):
        print(f"Training data reconstruction errors (manual vs built-in by sklearn) are not approx. equal, please check.")
        
    weightedSubgraphs_Test = nmfModel_FitTrain.transform(testDataHere)
    reconstructionError_Test = _beta_divergence(testDataHere,weightedSubgraphs_Test,subgraphExpressions,betaLossHere,square_root=True)
    print(f"Reconstruction error test data: {reconstructionError_Test}")
    reconstitutedArr_Test = np.matmul(weightedSubgraphs_Test,subgraphExpressions).copy()

    transformed_TrainData = nmfModel_FitTrain.transform(trainDataHere)
    transformed_TestData = nmfModel_FitTrain.transform(testDataHere)
    inverseTransform_TrainData = nmfModel_FitTrain.inverse_transform(transformed_TrainData)
    inverseTransform_TestData = nmfModel_FitTrain.inverse_transform(transformed_TestData)
    
    # Mantel r: reconstructed FC and original FC  
    
    fcReconstructed_filled_NMF_train = np.zeros((nRegions,nRegions,nRuns,nTrainSubjs)) # restFC_run1_filled_nmf_all_train
    fcReconstructed_filled_NMF_test = np.zeros((nRegions,nRegions,nRuns,nTestSubjs))
    
    # Reconstruct FC from training set based on NMF decomposition
    for trainSubjNum in range(nTrainSubjs):
        for runIx in range(nRuns):
            subjFC = np.zeros((nRegions,nRegions)) # temporary
            upperTriuIxs = np.triu_indices(len(subjFC),k=1)
            
            subjFC[upperTriuIxs] = inverseTransform_TrainData[:,int(trainSubjNum + (nTrainSubjs * runIx))]
            subjFC = subjFC + subjFC.T - np.diag(np.diag(subjFC))
            
            fcReconstructed_filled_NMF_train[:,:,runIx,trainSubjNum] = subjFC.copy()
            
    # Reconstruct FC from test set based on NMF decomposition
    for testSubjNum in range(nTestSubjs):
        for runIx in range(nRuns):
            subjFC = np.zeros((nRegions,nRegions)) # temporary
            upperTriuIxs = np.triu_indices(len(subjFC),k=1)
            
            subjFC[upperTriuIxs] = inverseTransform_TestData[:,int(testSubjNum + (nTestSubjs * runIx))]
            subjFC = subjFC + subjFC.T - np.diag(np.diag(subjFC))
            
            fcReconstructed_filled_NMF_test[:,:,runIx,testSubjNum] = subjFC.copy()            
            
    # NOTE: you have to pre-average across participants -- instead of doing mantel per participant then averaging -- because the participants are 
    # different across train and test sets. There are possibly more sophisticated ways to accomplish this at a "per subject" level (e.g., crossing 
    # each train and test subject then using an average), but it will add computation time and likely not change the final result much (fixed vs random 
    # effect sizes likely not hugely different here, but could run some tests if need be).
    
    trainAvg_FC_nmf = np.nanmean(fcReconstructed_filled_NMF_train,axis=3).copy()
    testAvg_FC_nmf = np.nanmean(fcReconstructed_filled_NMF_test,axis=3).copy()
    
    trainAvg_FC = np.arctanh(np.nanmean(np.tanh(fcData_Train),axis=3)).copy()
    testAvg_FC = np.arctanh(np.nanmean(np.tanh(fcData_Test),axis=3)).copy()
    
    # Run mantel tests
    # NOTE: for the arrays below, can set to np.zeros((3,nRuns)) and un-silence pVal etc. --> row ix 1 = mantel score (r-like), ix 2 = p-value, ix 3 = z-score
    
    # NOTE: these are all variants one might be interested in, but we recommend the following: 
    # Maximal mantelAll_TestOrig_TestInv (over parameter sweep and average over nBoots)
    # Compare mantelAll_TrainOrig_TestOrig and mantelAll_TrainInv_TestInv (are the 2 r-values close?)
    
    mantelAll_TrainOrig_TestOrig = np.zeros((nRuns)) # reference: train-test similarity, original FC arrays 
    mantelAll_TrainOrig_TrainInv = np.zeros((nRuns)) # reference: train-train similarity, original FC and NMF-FC
    mantelAll_TrainOrig_TestInv = np.zeros((nRuns)) # similarity of train-original-FC and test-NMF-FC
    mantelAll_TestOrig_TrainInv = np.zeros((nRuns)) # similarity of test-original-FC and train-NMF-FC
    mantelAll_TestOrig_TestInv = np.zeros((nRuns)) # similarity of test-original-FC and test-NMF-FC
    mantelAll_TrainInv_TestInv = np.zeros((nRuns)) # similarity of train-NMF-FC and test-NMF-FC 
    
    # NOTE: to use the mantel toolbox I linked above, need to 0 out diagonal and round to a reasonable decimal place (6 or 7), otherwise 
    # will get "ValueError: X is not a valid condensed or redundant distance matrix"
    for runIx in range(nRuns):
        xHere = trainAvg_FC[:,:,runIx].copy()
        yHere = testAvg_FC[:,:,runIx].copy()
        np.fill_diagonal(xHere, 0)
        np.fill_diagonal(yHere, 0)
        xHere = np.round(xHere,6)
        yHere = np.round(yHere,6)
        mantelScore, pVal, zVal = mantel.test(xHere,yHere, perms=1) # reference: train-test similarity, original FC arrays 
        mantelAll_TrainOrig_TestOrig[runIx] = mantelScore
        #mantelAll_TrainOrig_TestOrig[1,runIx] = pVal
        #mantelAll_TrainOrig_TestOrig[2,runIx] = zVal

        xHere = trainAvg_FC[:,:,runIx].copy()
        yHere = trainAvg_FC_nmf[:,:,runIx].copy()
        np.fill_diagonal(xHere, 0)
        np.fill_diagonal(yHere, 0)
        xHere = np.round(xHere,6)
        yHere = np.round(yHere,6)
        mantelScore, pVal, zVal = mantel.test(xHere,yHere, perms=1) # reference: train-train similarity, original FC and NMF-FC
        mantelAll_TrainOrig_TrainInv[runIx] = mantelScore
        #mantelAll_TrainOrig_TrainInv[1,runIx] = pVal
        #mantelAll_TrainOrig_TrainInv[2,runIx] = zVal
        
        xHere = trainAvg_FC[:,:,runIx].copy()
        yHere = testAvg_FC_nmf[:,:,runIx].copy()
        np.fill_diagonal(xHere, 0)
        np.fill_diagonal(yHere, 0)
        xHere = np.round(xHere,6)
        yHere = np.round(yHere,6)
        mantelScore, pVal, zVal = mantel.test(xHere,yHere, perms=1) # similarity of train-original-FC and test-NMF-FC
        mantelAll_TrainOrig_TestInv[runIx] = mantelScore
        #mantelAll_TrainOrig_TestInv[1,runIx] = pVal
        #mantelAll_TrainOrig_TestInv[2,runIx] = zVal
        
        xHere = testAvg_FC[:,:,runIx].copy()
        yHere = trainAvg_FC_nmf[:,:,runIx].copy()
        np.fill_diagonal(xHere, 0)
        np.fill_diagonal(yHere, 0)
        xHere = np.round(xHere,6)
        yHere = np.round(yHere,6)
        mantelScore, pVal, zVal = mantel.test(xHere,yHere, perms=1) # similarity of test-original-FC and train-NMF-FC
        mantelAll_TestOrig_TrainInv[runIx] = mantelScore
        #mantelAll_TestOrig_TrainInv[1,runIx] = pVal
        #mantelAll_TestOrig_TrainInv[2,runIx] = zVal
        
        xHere = testAvg_FC[:,:,runIx].copy()
        yHere = testAvg_FC_nmf[:,:,runIx].copy()
        np.fill_diagonal(xHere, 0)
        np.fill_diagonal(yHere, 0)
        xHere = np.round(xHere,6)
        yHere = np.round(yHere,6)
        mantelScore, pVal, zVal = mantel.test(xHere,yHere, perms=1) # similarity of test-original-FC and test-NMF-FC
        mantelAll_TestOrig_TestInv[runIx] = mantelScore
        #mantelAll_TestOrig_TestInv[1,runIx] = pVal
        #mantelAll_TestOrig_TestInv[2,runIx] = zVal
        
        xHere = trainAvg_FC_nmf[:,:,runIx].copy()
        yHere = testAvg_FC_nmf[:,:,runIx].copy()
        np.fill_diagonal(xHere, 0)
        np.fill_diagonal(yHere, 0)
        xHere = np.round(xHere,6)
        yHere = np.round(yHere,6)
        mantelScore, pVal, zVal = mantel.test(xHere,yHere, perms=1) # similarity of train-NMF-FC and test-NMF-FC 
        mantelAll_TrainInv_TestInv[runIx] = mantelScore
        #mantelAll_TrainInv_TestInv[1,runIx] = pVal
        #mantelAll_TrainInv_TestInv[2,runIx] = zVal
        
    return reconstructionError_Train, reconstructionError_Test, mantelAll_TrainOrig_TestOrig, mantelAll_TrainOrig_TrainInv, mantelAll_TrainOrig_TestInv, mantelAll_TestOrig_TrainInv, mantelAll_TestOrig_TestInv, mantelAll_TrainInv_TestInv

########################################################################################
def NMF_subgraph_stability(nmfInputArray_Train,
                           nmfInputArray_Test,
                           fcData_Train,
                           fcData_Test,
                           nRuns,
                           bootNum,
                           alphaHere_W,
                           betaLossHere,
                           nSubgraphs):
    ''' 
    After tuning alpha and beta parameters in NMF_cv() above, find stable number of subgraphs (k parameter, also called coefficients) 
    
    All inputs and notes are the same as NMF_cv() (see function above this one), except you need to specifcy nSubgraphs. We recommend an outer loop varying from 2-20 subgraphs.
    
    OUTPUTS: 
        reconstructionError_Test_k: standard accuracy for this k solution; should be minimized
        mantelAll_TestOrig_TestInv_k: FC-reconstruction based accuracy for this k solution; should be maximized
        
    '''
    
    _, reconstructionError_Test_k,_,_,_,_, mantelAll_TestOrig_TestInv_k,_ = NMF_cv(nmfInputArray_Train,
                                                                                   nmfInputArray_Test,
                                                                                   fcData_Train,
                                                                                   fcData_Test,
                                                                                   nRuns,
                                                                                   bootNum,
                                                                                   alphaHere_W,
                                                                                   betaLossHere,
                                                                                   nSubgraphs=nSubgraphs)
    
    return reconstructionError_Test_k, mantelAll_TestOrig_TestInv_k

########################################################################################
def NMF_validation(nmfInputArray,
                   nRuns,
                   nRegions,
                   alphaHere_W,
                   betaLossHere,
                   nSubgraphs,
                   l1RatioHere=0,
                   initHere=None,
                   solverHere='mu',
                   tolHere=0.0001,
                   maxIterHere=600,
                   randomStateHere=None,
                   alphaHere_H='same',
                   verboseHere=0,
                   shuffleHere=False):

    '''
    Once alpha, beta, and k parameters are tuned, run NMF on held-out validation set. Can format validation FC data with NMF_format_data() above.
    
    INPUTS: 
        nmfInputArray: output of NMF_format_data(), validation set that was held out. regional pairs x states & training subjects
        nRuns: number of runs/states in NMF input array (same as in NMF_format_data() and dimension 3 in fcData)
        nRegions: number of brain regions, or x and y axes of fcData used in NMF_format_data()
        alphaHere_W: value of alpha; regularization constant 
        betaLossHere: value of beta; beta divergence to be minimized 
        nSubgraphs: number of coefficients or k 
        l1RatioHere: initial tests show >0 overly sparsifies; so this is actually L2 ratio (when set to 0). Default 0
        initHere: default to None
        solverHere: may consider other options; note coordinate descent (cd) is sklearn default, but mu (multiplicative) is needed to sweep beta parameter. Default 'mu'
        tolHere: tolerance of stopping condition. Default 0.0001
        maxIterHere: iterations before timeout, 200 is sklearn default; I increased it based on warnings printed out --> Default 600
        randomStateHere: used only with init; or with coordinate descent. Default None
        alphaHere_H: Default 'same' --> use same alpha as sweeped by CV (for W matrix) for H matrix
        verboseHere: can set to True (1) if need to debug; Default 0 (False)
        shuffleHere: Used if you need to randomize coordiante order in solver; Default False 

    OUTPUTS: 
        weightedSubgraphs: features matrix (basis set) 
        subgraphExpressions: coefficients matrix (expressions; encoding matrix)
        weightedSubgraphs_NetShape: reformatting features matrix into k (nSubgraphs) number of region x region matrices (i.e., back into original brain network format)
        FC_All_Reconstituted: NMF-reconstituted input array 
    '''
    
    nmfModel = sklearn.decomposition.NMF(n_components=nSubgraphs, 
                                         init=initHere, 
                                         solver=solverHere, 
                                         beta_loss=betaLossHere, 
                                         tol=tolHere, 
                                         max_iter=maxIterHere, 
                                         random_state=randomStateHere, 
                                         alpha_W=alphaHere_W, 
                                         alpha_H=alphaHere_H, 
                                         l1_ratio=l1RatioHere, 
                                         verbose=verboseHere, 
                                         shuffle=shuffleHere)
    
    weightedSubgraphs = nmfModel.fit_transform(nmfInputArray)
    subgraphExpressions = nmfModel.components_
    
    weightedSubgraphs_NetShape = np.zeros((nRegions,nRegions,nSubgraphs))
    for subGraphNum in range(nSubgraphs):
        thisGraph_NetShape = np.zeros((nRegions,nRegions))
        thisGraph_NetShape[np.triu_indices(nRegions,k=1)] = weightedSubgraphs[:,subGraphNum].flatten()
        weightedSubgraphs_NetShape[:,:,subGraphNum] = thisGraph_NetShape.copy()

    FC_All_Reconstituted = np.matmul(weightedSubgraphs,subgraphExpressions)
    
    return weightedSubgraphs, subgraphExpressions, weightedSubgraphs_NetShape, FC_All_Reconstituted