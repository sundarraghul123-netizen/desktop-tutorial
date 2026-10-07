def quality_report(df):
    rows=[]
    for c in df.columns:
        missing=int(df[c].isna().sum())
        rows.append({'column':c,'dtype':str(df[c].dtype),'missing':missing,'missing_pct':round(missing/len(df)*100,2),'unique':int(df[c].nunique(dropna=True))})
    missing=int(df.isna().sum().sum()); dup=int(df.duplicated().sum())
    missing_penalty=min(35, missing/max(1,len(df)*len(df.columns))*100)
    dup_penalty=min(20, dup/max(1,len(df))*100)
    score=round(max(0,100-missing_penalty-dup_penalty),1)
    return {'missing_cells':missing,'duplicate_rows':dup,'score':score,'summary':{'rows':len(df),'columns':len(df.columns),'missing_cells':missing,'duplicate_rows':dup,'score':score},'columns':__import__('pandas').DataFrame(rows)}
