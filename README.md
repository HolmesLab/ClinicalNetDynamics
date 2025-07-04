# ClinicalNetDynamics

Code supporting: Brain network dynamics reflect psychiatric illness status and transdiagnostic symptom profiles across health and disease. Cocuzza C.V.*, Chopra S., Segal, A., Labache, L., Chin, R., Joss, K., and Holmes, A.J. (2025). 
In bioRxiv (p. 2025.05.23.655864). https://doi.org/10.1101/2025.05.23.655864

To investigate brain network dynamics linked with dimensionally-based symptom profiles exhibited across a transdiagnostic cohort of participants with and without psychiatric diagnoses. 

**Corresponding author email:** carrisacocuzza@gmail.com

**Repository contents:**
* Note that all scripts below include detailed annotations throughout; other contextual details may be found in the Methods section of the manuscript
  
* NMF_Cocuzza.py: python functions to implement non-negative matrix factorization (approach used to quantify brain network reconfiguration dynamics)
  
* Fingerprints_Cocuzza.py: python functions relevant to our symptom profiling/fingerprinting pipeline (note: RStudio used in select steps; notes are included where appropriate)
  
* Data_Splitting_Cocuzza.py: python script on how we split data into train/test/validation to avoid data leakage (see manuscript Methods)
  
* Heatmap_NetColors_Cocuzza.py: python function to visualize network color labels on x/y axes of functional connectivity matrices
  
* .Rmd scripts: RStudio scripts that are helpful for some functions inside Fingerprints_Cocuzza.py. See usage notes in Fingerprints_Cocuzza.py for details. Note that these R scripts require adaptation to your machine (e.g., directories, etc.) and research study (e.g., dataset specifications).
  
* .npy, .csv, and .pkl files. These are helper files for running functions in Fingerprints_Cocuzza.py (see usage notes in that script). These are fully de-identified and can be reconstituted by openly available Transdiagnostic Connectome Project data (see below for links to OpenNeuro and NDA). 

**Outside resources relevant to manuscript:**
* Transdiagnostic Connectome Project data via [OpenNeuro](https://openneuro.org/datasets/ds005237) and [NIMH Data Archive](https://nda.nih.gov/study.html?id=2932)
* [Transdiagnostic Connectome Project code repository](https://github.com/HolmesLab/TransdiagnosticConnectomeProject) (including resources for the pre-processing pipeline used in the present manuscript)
* [Cortical parcellation repository](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Yan2023_homotopic) (note: 400 parcel resolution used in mansucript as well as 17 networks per Yeo et al. 2011), [Yan et al., 2023, NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811923001568?via%3Dihub)
* [Subcortical atlas repository](https://github.com/yetianmed/subcortex?tab=readme-ov-file) (note: scale II used in manuscript), [Tian et al., 2021, Nature Neuroscience](https://www.nature.com/articles/s41593-020-00711-6.epdf?sharing_token=Fzk9fg_oTs49l2_4GcFHvtRgN0jAjWel9jnR3ZoTv0OcoEh_rWSSGTYcOuTVFJlvyoz7cKiJgYmHRlYIGzAnNt5tMyMZIXn3xdgdMC_wzDAONIDh5m0cUiLGzNChnEK_AHqVJl2Qrno8-hzk8CanTnXjGX3rRfZX3WXgTLew1oE%3D)
* [Cerebellum identification repository](https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/Buckner_2011) (note: see Buckner study for details on spatial autocorrelation regression), [Buckner et al., 2011, Journal of Neurophysiology](https://pubmed.ncbi.nlm.nih.gov/21795627/)
* Brain Connectivity Toolbox (used in select analyses; see network efficiency and participation coefficient) for [MATLAB](https://sites.google.com/site/bctnet/) and [Python](https://pypi.org/project/bctpy/)
* [Human Connectome Project Workbench](https://www.humanconnectome.org/software/connectome-workbench) for projecting results onto cortical surfaces (i.e., brain visualizations)
