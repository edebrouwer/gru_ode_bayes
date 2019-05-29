# GRU-ODE-Bayes

This implementation completes the paper : GRU-ODE-Bayes : continuous modeling of sporadically-observed time series.

Modeling real-world multidimensional time series can be particularly challenging when these are *sporadically* observed (i.e., sampling is irregular both in time and across dimensions)---such as in the case of clinical patient data. To address these challenges, we propose (1) a continuous-time version of the Gated Recurrent Unit, building upon the recent Neural Ordinary Differential Equations (Chen et al. 2018), and (2) a Bayesian update network that processes the sporadic observations. We bring these two ideas together in our GRU-ODE-Bayes method. 

This repository provides pytorch implementation of the GRU-ODE-Bayes paper. 

## Installation

### Requirements

The code uses Python3 and Pytorch as auto-differentiation package. The following python packages are required and will be automatically downloaded when installing the gru_ode_bayes package:

```
numpy
pandas
sklearn
torch
tensorflow (for logging)
tqdm
argparse
```

### Procedure

Install the main package :

```
pip install -e . 
```
And also the ODE numerical integration package : 
```
cd torchdiffeq
pip install -e .
```
## Run experiments
Experiments folder contains different cases study for the GRU-ODE-Bayes. Each trained model is then stored in the `trained_models` folder.
### 2-D Ornstein-Uhlenbeck SDE
Once in the double_OU folder, you can visualize some predictions of a previously trained model on newly generated data by running : 
```
cd experiments/double_OU
python double_ou_gruode.py --demo
```
This will print 10 new realizations of the process along with the model predictions.

For retraining the full model, run:
```
python double_ou_gruode.py
```
### Brusselator SDE
Similarly as for the 2D OU process, 
```
cd experiments/Brusselator
python run_gruode.py --demo 
```
will plot 10 new realizations of the process along with the model predictions. For retraining the full model :
```
python run_gruode.py
```

### USHCN daily (climate) data
For retraining the model, go to Climate folder and run 
```
python climate_gruode.py
```

## Datasets
The datasets for Brusselator, double OU and processed USHCN data have been uploaded on the repo for compatibility. 
The MIMIC dataset is not directly publicly available and was thus not pushed. It can be downloaded at : https://physionet.org/physiobank/database/mimic3cdb/

Folder 'data_preproc' contains all steps taken to preprocess the Climate and MIMIC datasets.

## Acknowledgements and References

The torchdiffeq package has been extended from the original version proposed by (Chen et al. 2018)

Chen et al. Neural ordinary differential equations, NeurIPS, 2018.

For climate dataset : 

Menne et al., Long-Term Daily Climate Records from Stations Across the Contiguous United States

