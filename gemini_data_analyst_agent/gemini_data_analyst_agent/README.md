# Hugging Face Professional Data Analyst Agent

## Windows / VS Code

```powershell
cd path\to\gemini_data_analyst_agent
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and add a Hugging Face token with Inference Providers permission:

```env
HF_TOKEN=your_huggingface_token
HF_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

Create a token at https://huggingface.co/settings/tokens with the `Make calls to Inference Providers` permission.

Run:

```powershell
streamlit run app.py
```

Upload CSV/XLSX/XLS. The application performs profiling, quality checks, EDA, KPIs, correlations, outliers, trends, SQL, Hugging Face planning/explanation, charts and reports.

## Architecture

Python/Pandas/DuckDB = verified calculations.
Hugging Face Inference Providers = planning, reasoning and explanation.
Plotly = interactive charts.
Streamlit = dashboard.
