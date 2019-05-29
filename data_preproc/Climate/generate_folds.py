import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

N = pd.read_csv("../processed_csv/small_chunked_sporadic.csv")["ID"].nunique()
full_idx = np.arange(N)
Nfolds = 5

np.random.seed(432)

for fold in range(Nfolds):
    train_idx, test_idx = train_test_split(np.arange(N),test_size=0.1)
    train_idx, val_idx = train_test_split(train_idx,test_size=0.2)
    fold_dir = f"small_chunk_fold_idx_{fold}/"
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    np.save(fold_dir+f"train_idx.npy",train_idx)
    np.save(fold_dir+f"val_idx.npy",val_idx)
    np.save(fold_dir+f"test_idx.npy",test_idx)
