import pandas as pd

def detect_date_column(df):
    dates=df.select_dtypes(include=['datetime','datetimetz']).columns.tolist()
    if dates:return dates[0]
    for c in df.select_dtypes(include=['object','string']).columns:
        sample=df[c].dropna().head(200)
        if len(sample)>=20:
            parsed=pd.to_datetime(sample,errors='coerce')
            if parsed.notna().mean()>=.9:return c
    return None

def trend_table(df,date_col):
    if not date_col:return pd.DataFrame()
    x=df.copy(); x[date_col]=pd.to_datetime(x[date_col],errors='coerce'); x=x.dropna(subset=[date_col])
    nums=x.select_dtypes(include='number').columns.tolist()
    if not nums:return pd.DataFrame()
    metric=next((c for c in nums if any(k in c.lower() for k in ['sales','revenue','profit','amount','income','value'])),nums[0])
    x['period']=x[date_col].dt.to_period('M').astype(str)
    out=x.groupby('period',as_index=False)[metric].sum().rename(columns={metric:'value'})
    out['pct_change']=out['value'].pct_change()*100
    return out
