from io import BytesIO
import pandas as pd

def build_excel(df,profile,quality,kpis,num,cat,corr,outliers,trend):
    b=BytesIO()
    with pd.ExcelWriter(b,engine='xlsxwriter') as w:
        pd.DataFrame([profile]).to_excel(w,sheet_name='Profile',index=False)
        pd.DataFrame([quality['summary']]).to_excel(w,sheet_name='Quality',index=False)
        pd.DataFrame(list(kpis.items()),columns=['KPI','Value']).to_excel(w,sheet_name='KPIs',index=False)
        num.to_excel(w,sheet_name='Statistics',index=False); cat.to_excel(w,sheet_name='Categories',index=False); corr.to_excel(w,sheet_name='Correlation'); outliers.to_excel(w,sheet_name='Outliers',index=False); trend.to_excel(w,sheet_name='Trends',index=False); df.head(1000).to_excel(w,sheet_name='Data_Sample',index=False)
    return b.getvalue()
