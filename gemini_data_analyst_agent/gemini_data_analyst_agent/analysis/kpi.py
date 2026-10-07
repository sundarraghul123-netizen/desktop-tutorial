import pandas as pd

def _fmt(v):
    if abs(v)>=1_000_000:return f'{v/1_000_000:.2f}M'
    if abs(v)>=1_000:return f'{v/1_000:.2f}K'
    return f'{v:,.2f}'

def discover_kpis(df):
    cols={c.lower():c for c in df.columns}; out={}
    def find(words):
        for w in words:
            for low,orig in cols.items():
                if w in low:return orig
    sales=find(['revenue','sales','amount']); profit=find(['profit','net income']); qty=find(['quantity','qty']); orders=find(['order_id','orderid','orders','transaction_id','invoice_id'])
    if sales: out['Total Revenue/Sales']=_fmt(pd.to_numeric(df[sales],errors='coerce').sum())
    if profit: out['Total Profit']=_fmt(pd.to_numeric(df[profit],errors='coerce').sum())
    if sales and profit:
        s=pd.to_numeric(df[sales],errors='coerce').sum(); p=pd.to_numeric(df[profit],errors='coerce').sum(); out['Profit Margin']=f'{(p/s*100):.2f}%' if s else 'N/A'
    if qty: out['Total Quantity']=_fmt(pd.to_numeric(df[qty],errors='coerce').sum())
    if orders: out['Orders']=f'{df[orders].nunique(dropna=True):,}'
    return out
