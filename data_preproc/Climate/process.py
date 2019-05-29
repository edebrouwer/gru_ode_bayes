import pandas as pd
from itertools import chain
import numpy as np
import re
import os



def to_pandas(state_dir):
    sep_list=[0,6,10,12,16]
    for day in range(31):
        sep_list = sep_list + [21+day*8,22+day*8,23+day*8,24+day*8]

    columns = ['COOP_ID','YEAR','MONTH','ELEMENT']
    values_list = list(chain.from_iterable(("VALUE-"+str(i+1),"MFLAG-"+str(i+1),"QFLAG-"+str(i+1),"SFLAG-"+str(i+1)) for i in range(31)))
    columns +=values_list

    df_list = []
    with open("./"+state_dir) as f:
        for line in f:
            line = line.strip()
            nl = [line[sep_list[i]:sep_list[i+1]] for i in range(len(sep_list)-1)]
            df_list.append(nl)
       
    df = pd.DataFrame(df_list,columns = columns)
    val_cols=[s for s in columns if "VALUE" in s]
    flag_cols = [s for s in columns if "FLAG" in s]
    mflag_cols = [s for s in columns if "MFLAG" in s]
    qflag_cols = [s for s in columns if "QFLAG" in s]
    sflag_cols = [s for s in columns if "SFLAG" in s]



    df[val_cols] = df[val_cols].astype(np.float32)

    df.replace(r'\s+',np.nan,regex=True,inplace = True)
    df.replace(-9999,np.nan,inplace=True)

    df_m = df.melt(id_vars = ["COOP_ID","YEAR","MONTH","ELEMENT"])
    df_m[["TYPE","DAY"]] = df_m.variable.str.split(pat="-",expand=True)

    df_n = df_m[["COOP_ID","YEAR","MONTH","DAY","ELEMENT","TYPE","value"]].copy()

    df_p = df_n.pivot_table(values = 'value', index = ["COOP_ID","YEAR","MONTH","DAY","ELEMENT"],columns = "TYPE", aggfunc="first")
    df_p.reset_index(inplace=True)

    df_q = df_p[["COOP_ID","YEAR","MONTH","DAY","ELEMENT","MFLAG","QFLAG","SFLAG","VALUE"]]

    name = state_dir[:-4]+'.csv'
    df_q.to_csv(name,index=False)
    #Number of non missing
    #meas_tot = df.shape[0]*31
    #na_meas = df[val_cols].isna().sum().sum()


    #mask_unvalid = df[qflag_cols].isin(["D","I","W","G","K","L","N","O","R","S","T","X","Z"])

def merge_dfs(target_dir = "./"):
    df_list = os.listdir(target_dir)
    csv_list = [s for s in f_list if '.csv' in s]
    state_csv_list = [s for s in csv_list if 'state' in s]


    df_list = []
    for state_csv in state_csv_list:
        print(f"Loading dataframe for state : {state_csv[:-4]}")
        df_temp = pd.read_csv(state_csv)
        df_temp.insert(0,"STATE",state_csv[-6:-4])
        df_list.append(df_temp)
    print("All dataframes are loaded")
    #Merge all datasets:
    print("Concat all ...")
    df = pd.concat(df_list)
    df.to_csv("daily_merged.csv",index=False)

def convert_all_to_pandas(target_dir = "./"):
    list_dir = os.listdir(target_dir)
    txt_list_dir = [s for s in list_dir if ".txt" in s]
    state_list_dir = [s for s in txt_list_dir if "state" in s]

    for state_dir in state_list_dir:
        print(f"Computing State : {state_dir}...")
        to_pandas(state_dir)

def clean_and_subsample():

    df =  pd.read_csv("daily_merged.csv")
    print(f"Loaded df. Number of observations : {df.shape[0]}")
    #Remove NaNs
    df.drop(df.loc[df.VALUE.isna()].index,inplace=True)

    #Remove values with bad quality flag.
    qflags = ["D","G","I","K","L","M","N","O","R","S","T","W","X","Z"]
    df.drop(df.loc[df.QFLAG.isin(qflags)].index,inplace=True)
    print(f"Removed bad quality flags. Number of observations {df.shape[0]}")

    #Check quality of measurements.
    df.loc[df.ELEMENT=="SNOW","VALUE"].min()
    df.loc[df.ELEMENT=="SNOW","VALUE"].max()

    df.loc[df.ELEMENT=="PRCP","VALUE"].min()
    df.loc[df.ELEMENT=="PRCP","VALUE"].max()

    df.loc[df.ELEMENT=="SNWD","VALUE"].min()
    df.loc[df.ELEMENT=="SNWD","VALUE"].max()

    df.loc[df.ELEMENT=="TMAX","VALUE"].min()
    df.loc[df.ELEMENT=="TMAX","VALUE"].max()

    df.loc[df.ELEMENT=="TMIN","VALUE"].min()
    df.loc[df.ELEMENT=="TMIN","VALUE"].max()

    #Compute number of measurements per center
    obs_ratio = df.shape[0]/df.COOP_ID.nunique()

    #Crop centers with no observations after 2001
    gp_id_year = df.groupby("COOP_ID")["YEAR"]
    min_max_year = gp_id_year.max().min()
    prop_over_2000 = (gp_id_year.max()>2001).sum()/gp_id_year.max().shape[0]
    print(f"Remove centers with no measurements after 2001")
    coop_list = list(gp_id_year.max().loc[gp_id_year.max()>2001].index)
    df.drop(df.loc[~df.COOP_ID.isin(coop_list)].index,inplace = True)
    # Crop center with no observations below 1970
    gp_id_year = df.groupby("COOP_ID")["YEAR"]
    max_min_year = gp_id_year.min().max()
    prop_over_1950 = (gp_id_year.min()<1970).sum()
    crop_list = list(gp_id_year.min().loc[gp_id_year.min()<1970].index)
    df.drop(df.loc[~df.COOP_ID.isin(crop_list)].index,inplace=True)

    #Crop the observations below 1950 and after 2001.
    df = df.loc[df.YEAR>=1950].copy()
    df = df.loc[df.YEAR<=2000].copy()

    print(f"Number of kept centers : {df.COOP_ID.nunique()}")
    print(f"Number of observations / center : {df.shape[0]/df.COOP_ID.nunique()}")
    print(f"Number of days : {50*365}")

    sample_df = df.groupby(["COOP_ID","ELEMENT"]).apply(lambda x : x.sample(frac = 0.05))
    sample_df.reset_index(drop=True, inplace = True)

    print(f"Number of kept centers : {sample_df.COOP_ID.nunique()}")
    print(f"Number of observations / center : {sample_df.shape[0]/sample_df.COOP_ID.nunique()}")
    print(f"Number of days : {50*365}")
    print(f"Number of observations per day : {sample_df.shape[0]/(sample_df.COOP_ID.nunique()*50*365)}")

    #Create a unique_index
    unique_map = dict(zip(list(sample_df.COOP_ID.unique()),np.arange(sample_df.COOP_ID.nunique())))
    label_map = dict(zip(list(sample_df.ELEMENT.unique()),np.arange(sample_df.ELEMENT.nunique())))

    sample_df.insert(0,"UNIQUE_ID",sample_df.COOP_ID.map(unique_map))
    sample_df.insert(1,"LABEL",sample_df.ELEMENT.map(label_map))

    #Create a time_index.
    import datetime
    sample_df["DATE"] = pd.to_datetime((sample_df.YEAR*10000+sample_df.MONTH*100+sample_df.DAY).apply(str),format='%Y%m%d')
    sample_df["DAYS_FROM_1950"] = (sample_df.DATE-datetime.datetime(1950,1,1)).dt.days

    #Normalize values.
    for label in ["SNOW","SNWD","PRCP","TMAX","TMIN"]:
        avg = sample_df.loc[sample_df.ELEMENT==label,"VALUE"].mean()
        s_dev = sample_df.loc[sample_df.ELEMENT==label,"VALUE"].std()
        sample_df.loc[sample_df.ELEMENT==label,"VALUE"] -= avg
        sample_df.loc[sample_df.ELEMENT==label,"VALUE"] /= s_dev

    #GENERATE TIME STAMP !!
    sample_df["TIME_STAMP"] = 2500*sample_df.DAYS_FROM_1950/sample_df.DAYS_FROM_1950.max()
    sample_df["TIME_STAMP"] = sample_df["TIME_STAMP"].round(1)

    sample_df.to_csv("cleaned_and_subsampled_df.csv",index=False)

    #Save dict.
    np.save("centers_id_mapping.npy",unique_map)
    np.save("label_id_mapping.npy",label_map)

def make_fit_for_gru_ode():
    df = pd.read_csv("cleaned_and_subsampled_df.csv")
    unique_centers = df["UNIQUE_ID"].unique()
    nunique_labels  = df["LABEL"].nunique()
    labels_col_names = ["Value_"+str(n) for n in np.sort(df["LABEL"].unique())]
    mask_col_names   = ["Mask_"+str(n) for n in np.sort(df["LABEL"].unique())]
    col = ["ID"]+["Time"]+labels_col_names+mask_col_names

    import tqdm
    df_new = pd.DataFrame(columns = col)
    for center in tqdm.tqdm(list(df.UNIQUE_ID.unique())):
        df_center=df.loc[df["UNIQUE_ID"]==center]
        unique_times = df_center["TIME_STAMP"].unique()
        entry = np.zeros((len(unique_times),2*nunique_labels+2))
        for idx_time,time in enumerate(unique_times):
            idx = df_center.loc[df_center["TIME_STAMP"]==time,"LABEL"]
            val = df_center.loc[df_center["TIME_STAMP"]==time,"VALUE"]
            entry[idx_time,idx.values+2]=val.values
            entry[idx_time,idx.values+2+nunique_labels]=1
            entry[idx_time,1]= time
            entry[idx_time,0]= center
        df_new = df_new.append(pd.DataFrame(entry, columns = col))
    df_new.to_csv(f"daily_sporadic.csv",index=False)

def chunk_series():
    df = pd.read_csv("daily_sporadic.csv")
    df["initial_cent"] = df.ID
    num_centers = df.ID.nunique()
    max_id = df.ID.max()
    df_list = []
    for i in range(25):
        df_year = df.loc[((i*100)<=df.Time)&(df.Time<((i+1)*100))].copy()
        df_year.Time = df_year.Time-(i*100)
        df_year.ID = df_year.ID + i*max_id
        #assert num_centers == df_year.ID.nunique()
        df_list.append(df_year)
    df_chunked = pd.concat(df_list)
    #Remove ids with less than 3 observations after time 75.
    above_df = df_chunked.loc[df_chunked.Time>75]
    num_obs_series = above_df.groupby("ID")["Time"].nunique()
    valid_list = list(num_obs_series[num_obs_series>3].index)
    df_clean = df_chunked.loc[df_chunked.ID.isin(valid_list)]
    assert df_clean.loc[df_clean.Time>75,"ID"].nunique() == df_clean.ID.nunique()
    df_clean[['ID', 'Time', 'Value_0', 'Value_1', 'Value_2', 'Value_3','Value_4', 'Mask_0', 'Mask_1', 'Mask_2', 'Mask_3', 'Mask_4']].to_csv("chunked_sporadic.csv",index=False)

    df_small = df.loc[df.Time>=2300].copy()
    df_small.Time = df_small.Time-2300
    above_df = df_small.loc[df_small.Time>150]
    num_obs_series = above_df.groupby("ID")["Time"].nunique()
    valid_list = list(num_obs_series[num_obs_series>3].index)
    df_clean_small = df_small.loc[df_small.ID.isin(valid_list)]
    df_clean_small[['ID', 'Time', 'Value_0', 'Value_1', 'Value_2', 'Value_3','Value_4', 'Mask_0', 'Mask_1', 'Mask_2', 'Mask_3', 'Mask_4']].to_csv("small_chunked_sporadic.csv",index=False)

if __name__=="__main__":
    #convert_all_to_pandas()
    #merge_dfs()
    clean_and_subsample()
    make_fit_for_gru_ode()
    chunk_series()

