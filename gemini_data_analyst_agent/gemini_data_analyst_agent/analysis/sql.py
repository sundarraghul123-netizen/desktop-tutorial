import duckdb

def run_sql(df,sql):
    q=sql.strip().rstrip(';')
    low=q.lower()
    if not (low.startswith('select') or low.startswith('with')):raise ValueError('Only SELECT/WITH queries are allowed.')
    blocked=['insert','update','delete','drop','alter','create','copy','attach','install','load','pragma','export','import']
    if any(word in low.split() for word in blocked):raise ValueError('Unsafe SQL statement blocked.')
    return duckdb.query_df(df,'data',q).df()
