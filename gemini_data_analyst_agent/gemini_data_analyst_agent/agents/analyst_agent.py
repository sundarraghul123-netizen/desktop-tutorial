import json
import os
import re
import pandas as pd
import requests
from analysis.correlation import correlation_matrix
from analysis.eda import numeric_summary, categorical_summary
from analysis.outliers import outlier_report
from analysis.trends import detect_date_column, trend_table


class AnalystAgent:
    DEFAULT_MODELS = [
        'Qwen/Qwen3-4B-Instruct-2507',
        'google/gemma-3-4b-it',
        'meta-llama/Llama-3.1-8B-Instruct',
    ]

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HF_TOKEN')
        configured_model = model or os.getenv('HF_MODEL') or self.DEFAULT_MODELS[0]
        self.model = configured_model.strip()
        if not self.api_key:
            raise ValueError('Missing Hugging Face API token. Set HUGGINGFACE_API_TOKEN or HF_TOKEN in your environment.')

    def _extract_text(self, payload):
        if isinstance(payload, dict) and payload.get('choices'):
            message = payload['choices'][0].get('message', {})
            content = message.get('content', '') if isinstance(message, dict) else ''
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return ''.join(part.get('text', '') for part in content if isinstance(part, dict))

        if isinstance(payload, list):
            if not payload:
                return ''
            first = payload[0]
            if isinstance(first, dict):
                text = first.get('generated_text') or first.get('answer') or ''
                return text
            return str(first)

        if isinstance(payload, dict):
            return payload.get('generated_text') or payload.get('answer') or json.dumps(payload, default=str)

        return str(payload)

    def _call_hf(self, prompt, json_mode=False):
        candidates = []
        seen = set()
        for candidate in [self.model] + [m for m in self.DEFAULT_MODELS if m != self.model]:
            if candidate and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

        last_error = None
        last_error_status = None
        for model_name in candidates:
            url = 'https://router.huggingface.co/v1/chat/completions'
            payload = {
                'model': model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1000,
                'temperature': 0.1,
            }
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=180)
            except requests.RequestException as exc:
                raise RuntimeError(f'Could not reach Hugging Face Inference Providers: {exc}') from exc

            if response.status_code < 400:
                self.model = model_name
                return response.json()

            try:
                detail = response.json()
            except ValueError:
                detail = {'error': response.text}

            last_error = detail
            last_error_status = response.status_code
            if response.status_code in (404, 502, 503, 504):
                continue

            raise RuntimeError(self._format_hf_error(response.status_code, detail))

        if last_error is not None:
            if isinstance(last_error, dict):
                raise RuntimeError(self._format_hf_error(last_error_status, last_error))
            raise RuntimeError(f'Hugging Face request failed: {last_error}')

        raise RuntimeError('No configured Hugging Face model is available through Inference Providers. Try Qwen/Qwen3-4B-Instruct-2507 or another chat model listed at https://huggingface.co/models?inference_provider=all&pipeline_tag=conversational.')

    @staticmethod
    def _format_hf_error(status_code, detail):
        if isinstance(detail, dict):
            for key in ('error', 'message', 'detail'):
                value = detail.get(key)
                if isinstance(value, str) and value.strip():
                    return f'Hugging Face API error {status_code}: {value}'

            if detail.get('status'):
                return f'Hugging Face API error {status_code}: the requested model is unavailable or access is restricted ({detail.get("status")}).'

            if status_code in (502, 503, 504):
                return f'Hugging Face API error {status_code}: inference providers are temporarily unavailable for the configured models. Please retry shortly.'

            return f'Hugging Face API error {status_code}: the model is not available, private, or not supported by the Inference API.'
        return f'Hugging Face API error {status_code}: {detail}'

    def call(self, prompt, json_mode=False):
        response = self._call_hf(prompt, json_mode=json_mode)
        text = self._extract_text(response)
        if json_mode:
            text = text.strip()
            if text.startswith('```'):
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```\s*$', '', text)
            return text
        return text

    def evidence(self, df, profile, quality, kpis, trend, outliers):
        corr = correlation_matrix(df)
        return {'profile': profile, 'quality': quality['summary'], 'kpis': kpis, 'numeric': numeric_summary(df).round(4).to_dict('records'), 'categorical': categorical_summary(df).to_dict('records'), 'correlation': corr.round(4).to_dict() if not corr.empty else {}, 'outliers': outliers.round(4).to_dict('records'), 'trend': trend.round(4).to_dict('records') if not trend.empty else []}

    def executive_analysis(self, df, profile, quality, kpis, trend, outliers):
        ev = self.evidence(df, profile, quality, kpis, trend, outliers)
        p = f'''You are a senior business data analyst. Analyze ONLY the verified evidence below. Never invent numbers. Return professional Markdown with sections: Executive Summary, KPI Findings, Trends and Patterns, Data Quality Risks, Anomalies, Business Implications, Recommended Actions, Limitations. Clearly distinguish facts from hypotheses.\nEVIDENCE:\n{json.dumps(ev, default=str)[:40000]}'''
        return self.call(p)

    def plan(self, q, df, profile):
        schema = [{'name': c, 'dtype': str(df[c].dtype), 'sample': df[c].dropna().astype(str).head(3).tolist()} for c in df.columns]
        p = f'''You are a professional data analyst. Create a JSON plan for this question using only these columns. Operations allowed: groupby, top_n, trend, correlation, outliers, distribution, summary, filter. Question: {q}. Schema: {json.dumps(schema, default=str)} Return JSON keys operation, group_by, metric, column, n, chart. Do not invent columns.'''
        text = self.call(p, json_mode=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            cleaned = text[text.find('{'):text.rfind('}') + 1]
            return json.loads(cleaned)

    def execute_plan(self, plan, df):
        op = plan.get('operation'); group = plan.get('group_by'); metric = plan.get('metric'); col = plan.get('column'); n = int(plan.get('n') or 10); chart = plan.get('chart', 'none')
        if op == 'summary':
            data = numeric_summary(df); return {'evidence': data.to_dict('records')}
        if op == 'correlation':
            data = correlation_matrix(df); return {'evidence': data.round(4).to_dict()}
        if op == 'outliers':
            data = outlier_report(df); return {'evidence': data.round(4).to_dict('records')}
        if op == 'trend':
            d = detect_date_column(df); data = trend_table(df, d); return {'evidence': data.round(4).to_dict('records')}
        if op == 'distribution':
            if col not in df.columns:
                raise ValueError('Invalid column selected by AI.')
            data = df[col].describe().to_dict(); import plotly.express as px; fig = px.histogram(df, x=col, title=f'Distribution of {col}'); return {'evidence': data, 'chart': fig}
        if op in ('groupby', 'top_n'):
            if metric not in df.columns:
                raise ValueError('Invalid metric selected by AI.')
            if op == 'groupby':
                if group not in df.columns:
                    raise ValueError('Invalid group column selected by AI.')
                s = pd.to_numeric(df[metric], errors='coerce'); x = pd.DataFrame({group: df[group], metric: s}).dropna().groupby(group)[metric].sum().sort_values(ascending=False).head(n).reset_index()
                import plotly.express as px; fig = px.bar(x, x=group, y=metric, title=f'{metric} by {group}'); return {'evidence': x.to_dict('records'), 'chart': fig}
            x = df[[metric]].apply(pd.to_numeric, errors='coerce').dropna().sort_values(metric, ascending=False).head(n).reset_index(drop=True); import plotly.express as px; fig = px.bar(x, y=metric, title=f'Top {n} {metric} values'); return {'evidence': x.to_dict('records'), 'chart': fig}
        if op == 'filter':
            if col not in df.columns:
                raise ValueError('Invalid column selected by AI.')
            return {'evidence': df[[col]].dropna().head(100).to_dict('records')}
        raise ValueError('Unsupported analysis operation.')

    def explain(self, q, plan, result):
        ev = {k: v for k, v in result.items() if k != 'chart'}
        p = f'''You are a senior data analyst. Answer the question using ONLY this verified evidence. Never invent or recalculate values. Give a concise professional answer with Direct answer, Evidence, Business interpretation, Caveat. Question: {q}\nPlan: {json.dumps(plan, default=str)}\nEvidence: {json.dumps(ev, default=str)[:20000]}'''
        return self.call(p)
