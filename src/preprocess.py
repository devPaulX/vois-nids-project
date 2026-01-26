import pandas as pd
import numpy as np

def load_data(filepath, nrows=15000):
    df = pd.read_csv(filepath, nrows=nrows)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df