# Data pre-processing

## MIMIC pre-processing

### Procedure

#### Data Cleaning

*First run the whole Admissions notebook. This will process the Patients database, the admission database and the inputs database.
It will ouput `INPUTS_processed.csv` and `Admissions_processed.csv`.

*Then run Ouputs notebook. It will process the OUTPUTEVENTS database and output `OUTPUTS_processed.csv`

*Run LabEvents notebook. It will process the LABEVENTS database and output `LAB_processed.csv`

*Run Prescriptions notebook. It will process the PRESCRIPTIONS database and output `PRESCRIPTIONS_processed.csv`

*Those processed tables are merged together in the DataMerging notebook. This will output : `complete_tensor.csv`, `complete_death_tags.csv` and `complete_covariates.csv`.

#### Processing for GRU-ODE-Bayes

* Run `mimic_preproc.py` adapting the path to fit the location of your files. This will output 3 files : `Processed_MIMIC.csv`, `MIMIC_tags.csv`and `MIMIC_covs.csv`.

Tags and covs are mortality labels and covariates respectively, but thoses have'nt been used in the results shown in the paper.

### Folds generation.

For generation 5 train/validation/test folds on the MIMIC data, use folds_split_mimic.py





