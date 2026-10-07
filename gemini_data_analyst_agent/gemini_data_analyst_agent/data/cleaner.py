import pandas as pd

def clean_dataset(df):
    df = df.copy()
    log=[]
    before=len(df)
    df=df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if len(df)!=before: log.append(f'Removed {before-len(df)} completely empty rows.')
    old=list(df.columns); df.columns=[str(c).strip() for c in df.columns]
    if old!=list(df.columns): log.append('Trimmed whitespace from column names.')
    for c in df.select_dtypes(include=['object']).columns:
        s=df[c].astype('string')
        stripped=s.str.strip()
        if not stripped.equals(s): df[c]=stripped; log.append(f'Trimmed text whitespace in {c}.')
        # Conservative date inference
        sample=df[c].dropna().head(100)
        if len(sample)>=10:
            parsed=pd.to_datetime(sample, errors='coerce')
            if parsed.notna().mean()>=0.85:
                df[c]=pd.to_datetime(df[c], errors='coerce'); log.append(f'Inferred datetime column: {c}.')
    return df,log
