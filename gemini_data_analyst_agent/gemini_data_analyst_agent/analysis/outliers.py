import numpy as np
import pandas as pd

def outlier_report(df):
    rows=[]
    for c in df.select_dtypes(include=np.number).columns:
        s=df[c].dropna()
        if len(s)<4: continue
        q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr
        count=int(((s<lo)|(s>hi)).sum())
        rows.append({'column':c,'outliers':count,'outlier_pct':round(count/len(s)*100,2),'lower_bound':lo,'upper_bound':hi})
    return pd.DataFrame(rows)
