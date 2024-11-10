import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


root = "/Users/hamishburke/Documents/AI/datasets"
malware = "malware"

def load_dataset(rootdir=root,dataset=malware):
    full_path = os.path.join(rootdir,dataset)
    data_paths = os.listdir(full_path)

    csv_dfs = []

    for path in data_paths:
        if path.endswith("csv"): # malware delimiter is | instaed of ,
            csv_df = pd.read_csv(os.path.join(full_path,path), delimiter="|")
            csv_dfs.append(csv_df)
            print("Read ", path)
            break # line 47

    return csv_dfs


def plot_hist_for_features(df,path_to_save=None):
    pass



def get_unique_dict(df:pd.DataFrame):
    value_map = pd.DataFrame()

    # for each feature, get a list of the unique values
    for column in df.columns:
        df[column].plot.hist()
        values = df[column].unique() # series object
        value_map[column] = values

    # return map
    return value_map

def summary_stats(df:pd.DataFrame):
    for column in df.columns:
        hist = df[column].plot.hist()
        plt.show()

if __name__ == "__main__":
    dfs = load_dataset()

    ## stack into one
    all_data = dfs[0]
    for i in range(1,len(dfs)):
        # will skip this for now, only will use first file
        np_arr = np.asarray(dfs[i]) # not working, will skip until can seaerch up lol
        all_data = np.stack(all_data,np_arr) # have to convert to numpy array

    

    summary_stats(all_data)





    ## create dict of unique values for each feature
    unique_values = get_unique_dict(all_data)

    for column in unique_values.columns:
        values = unique_values[column]
        print(len(values)," unique values for ", column)



