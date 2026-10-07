import numpy as np
import pandas as pd

def numeric_summary(df):
    n=df.select_dtypes(include=np.number)
    if n.empty:return pd.DataFrame()
    out=n.describe().T.reset_index().rename(columns={'index':'column'})
    out['median']=n.median().values; out['skewness']=n.skew().values; out['kurtosis']=n.kurt().values
    return out

def categorical_summary(df):
    rows=[]
    for c in df.select_dtypes(include=['object','category','string']).columns:
        vc=df[c].value_counts(dropna=False).head(10)
        rows.append({'column':c,'unique':int(df[c].nunique(dropna=True)),'top_values':'; '.join(f'{k} ({v})' for k,v in vc.items())})
    return pd.DataFrame(rows)
