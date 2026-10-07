import pandas as pd

def correlation_matrix(df):
    n=df.select_dtypes(include='number')
    return n.corr() if n.shape[1]>=2 else pd.DataFrame()
