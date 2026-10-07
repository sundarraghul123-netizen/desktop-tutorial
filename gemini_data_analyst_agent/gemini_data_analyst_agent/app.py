import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from data.loader import load_dataset
from data.profiler import profile_dataset
from data.cleaner import clean_dataset
from analysis.quality import quality_report
from analysis.eda import numeric_summary, categorical_summary
from analysis.correlation import correlation_matrix
from analysis.outliers import outlier_report
from analysis.trends import detect_date_column, trend_table
from analysis.kpi import discover_kpis
from analysis.charts import make_chart
from analysis.sql import run_sql
from agents.analyst_agent import AnalystAgent
from reports.excel_report import build_excel
from reports.pdf_report import build_pdf

load_dotenv()
load_dotenv('.env.huggingface')
st.set_page_config(page_title='Hugging Face Data Analyst', page_icon='📊', layout='wide')

st.title('📊 Hugging Face Data Analyst Agent')
st.caption('Python verifies the numbers • Hugging Face reasons • Plotly visualizes • DuckDB queries')

with st.sidebar:
    st.header('⚙️ Configuration')
    api_key = st.text_input('Hugging Face API key', value=os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HF_TOKEN',''), type='password')
    model = st.text_input('Hugging Face model', value=os.getenv('HF_MODEL','Qwen/Qwen3-4B-Instruct-2507'))
    uploaded = st.file_uploader('Upload CSV / Excel', type=['csv','xlsx','xls'])

if not uploaded:
    st.info('Upload a CSV or Excel file to start.')
    st.markdown('''
### What this application does
- Automatic data profiling and quality audit
- EDA, statistics, correlations and outliers
- KPI discovery and trend analysis
- Interactive Plotly charts
- Read-only DuckDB SQL
- Hugging Face-powered professional analyst chat
- AI executive summary
- PDF and Excel report export
''')
    st.stop()

try:
    raw_df = load_dataset(uploaded)
    df, cleaning_log = clean_dataset(raw_df)
except Exception as e:
    st.error(f'File processing failed: {e}')
    st.stop()

profile = profile_dataset(df)
quality = quality_report(df)
kpis = discover_kpis(df)
num = numeric_summary(df)
cat = categorical_summary(df)
corr = correlation_matrix(df)
outliers = outlier_report(df)
date_col = detect_date_column(df)
trend = trend_table(df, date_col) if date_col else pd.DataFrame()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'executive' not in st.session_state:
    st.session_state.executive = ''

# ---------------- Dashboard ----------------
tabs = st.tabs(['📌 Executive Dashboard','🔎 EDA','🧹 Data Quality','📈 Trends','🤖 AI Analyst','🧮 SQL','📄 Reports'])

with tabs[0]:
    st.subheader('Executive Dashboard')
    cols = st.columns(4)
    cols[0].metric('Rows', f"{len(df):,}")
    cols[1].metric('Columns', f"{len(df.columns):,}")
    cols[2].metric('Missing cells', f"{quality['missing_cells']:,}")
    cols[3].metric('Duplicate rows', f"{quality['duplicate_rows']:,}")

    st.markdown('### Discovered KPIs')
    if kpis:
        kc = st.columns(min(4, len(kpis)))
        for i, (name, value) in enumerate(kpis.items()):
            kc[i % len(kc)].metric(name, value)
    else:
        st.info('No standard business KPI columns were detected.')

    c1,c2 = st.columns(2)
    with c1:
        if not trend.empty:
            st.plotly_chart(make_chart(trend, 'line', trend.columns[0], trend.columns[1], f'{trend.columns[1]} over time'), use_container_width=True)
        elif not num.empty:
            col = num.iloc[0]['column']
            st.plotly_chart(make_chart(df, 'histogram', col, col, f'Distribution of {col}'), use_container_width=True)
    with c2:
        if len(df.select_dtypes('number').columns) >= 2:
            a,b = df.select_dtypes('number').columns[:2]
            st.plotly_chart(make_chart(df, 'scatter', a, b, f'{b} vs {a}'), use_container_width=True)

    st.markdown('### AI Executive Analysis')
    if st.button('🧠 Generate executive analysis', type='primary'):
        if not api_key:
            st.warning('Enter your Hugging Face API key in the sidebar.')
        else:
            with st.spinner('Hugging Face is analyzing verified evidence...'):
                try:
                    agent = AnalystAgent(api_key, model)
                    st.session_state.executive = agent.executive_analysis(df, profile, quality, kpis, trend, outliers)
                except Exception as e:
                    st.error(f'Hugging Face error: {e}')
    if st.session_state.executive:
        st.markdown(st.session_state.executive)

with tabs[1]:
    st.subheader('Exploratory Data Analysis')
    st.markdown('### Numeric statistics')
    st.dataframe(num, use_container_width=True)
    st.markdown('### Categorical summary')
    st.dataframe(cat, use_container_width=True)
    st.markdown('### Correlation matrix')
    st.dataframe(corr.round(3) if not corr.empty else pd.DataFrame({'message':['Need at least two numeric columns']}), use_container_width=True)
    st.markdown('### Outlier audit')
    st.dataframe(outliers, use_container_width=True)

with tabs[2]:
    st.subheader('Data Quality Audit')
    st.metric('Quality score', f"{quality['score']}/100")
    st.dataframe(quality['columns'], use_container_width=True)
    if cleaning_log:
        st.markdown('### Cleaning actions')
        for item in cleaning_log:
            st.write('•', item)

with tabs[3]:
    st.subheader('Trend Analysis')
    if date_col:
        st.success(f'Detected date column: {date_col}')
        st.dataframe(trend, use_container_width=True)
        if not trend.empty:
            metric_col = trend.columns[1]
            st.plotly_chart(make_chart(trend, 'line', trend.columns[0], metric_col, f'{metric_col} trend'), use_container_width=True)
    else:
        st.info('No reliable date column was detected.')

with tabs[4]:
    st.subheader('Ask the Professional Data Analyst')
    examples = ['What are the top 10 categories by revenue?', 'What are the biggest data quality risks?', 'Which numeric columns are strongly correlated?', 'Why might sales be declining?', 'Give me five business recommendations.']
    st.caption('Examples: ' + ' | '.join(examples))
    for role,msg in st.session_state.chat_history:
        with st.chat_message(role): st.markdown(msg)
    question = st.chat_input('Ask a question about your dataset...')
    if question:
        st.session_state.chat_history.append(('user', question))
        with st.chat_message('user'): st.markdown(question)
        if not api_key:
            answer = 'Please enter your Gemini API key in the sidebar.'
            with st.chat_message('assistant'): st.warning(answer)
            st.session_state.chat_history.append(('assistant', answer))
        else:
            with st.chat_message('assistant'):
                with st.spinner('Planning, calculating and reasoning...'):
                    try:
                        agent = AnalystAgent(api_key, model)
                        plan = agent.plan(question, df, profile)
                        result = agent.execute_plan(plan, df)
                        answer = agent.explain(question, plan, result)
                        st.markdown(answer)
                        if result.get('chart') is not None:
                            st.plotly_chart(result['chart'], use_container_width=True)
                        with st.expander('Verified analysis evidence'):
                            st.json(result.get('evidence', {}))
                        st.session_state.chat_history.append(('assistant', answer))
                    except Exception as e:
                        st.error(f'Analysis failed: {e}')

with tabs[5]:
    st.subheader('Read-only SQL Explorer')
    st.caption('Table name: data. Only SELECT / WITH queries are allowed.')
    sql = st.text_area('SQL query', 'SELECT * FROM data LIMIT 20')
    if st.button('Run query'):
        try:
            st.dataframe(run_sql(df, sql), use_container_width=True)
        except Exception as e:
            st.error(str(e))

with tabs[6]:
    st.subheader('Professional Reports')
    executive = st.session_state.executive or 'Generate the AI executive analysis from the dashboard first.'
    c1,c2 = st.columns(2)
    with c1:
        if st.button('Generate PDF report'):
            path = build_pdf(df, profile, quality, kpis, executive)
            st.download_button('⬇️ Download PDF', open(path,'rb').read(), file_name='data_analyst_report.pdf', mime='application/pdf')
    with c2:
        if st.button('Generate Excel report'):
            data = build_excel(df, profile, quality, kpis, num, cat, corr, outliers, trend)
            st.download_button('⬇️ Download Excel', data, file_name='data_analyst_report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

with st.expander('Dataset preview'):
    st.dataframe(df.head(100), use_container_width=True)
