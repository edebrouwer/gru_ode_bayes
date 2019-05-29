# Data pre-processing

## USHCN pre-processing

### Procedure
* Download all USHCN text files in this directory. They should look like 'stateXX.txt'

* run 'python process.py'

This will convert all files to pandas format, apply the pre-processing and finally return a csv file : 'small_chunked_sporadic.csv'. This is the file needed to run GRU-ODE-Bayes. (See experiment section)


### Folds

For cross-validation, we provide a script that generates 5 folds for the data. Edit the first line of `generate_folds.py` to adapt it to the path of your local `small_chunked_sporadic.csv`. Then run the script, it will generate directories with train, validation and test folds (5folds each) in the `.npy` format.
