import numpy as np

def profile_dataset(df):
    return {
        'rows':len(df),'columns':len(df.columns),'memory_mb':round(df.memory_usage(deep=True).sum()/1024**2,2),
        'numeric':df.select_dtypes(include=np.number).columns.tolist(),
        'categorical':df.select_dtypes(include=['object','category','string']).columns.tolist(),
        'datetime':df.select_dtypes(include=['datetime','datetimetz']).columns.tolist(),
        'columns':[{'name':c,'dtype':str(df[c].dtype),'unique':int(df[c].nunique(dropna=True))} for c in df.columns]
    }
