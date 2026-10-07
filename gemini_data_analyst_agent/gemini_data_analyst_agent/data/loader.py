import pandas as pd

def load_dataset(uploaded):
    name = uploaded.name.lower()
    if name.endswith('.csv'):
        uploaded.seek(0)
        try: df = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0); df = pd.read_csv(uploaded, encoding='latin1')
    else:
        uploaded.seek(0); df = pd.read_excel(uploaded)
    if df.empty: raise ValueError('The uploaded dataset is empty.')
    df.columns = [str(c).strip() for c in df.columns]
    if any(not c for c in df.columns): raise ValueError('Blank column names are not allowed.')
    return df
