from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def build_pdf(df,profile,quality,kpis,executive):
    path=Path('reports/data_analyst_report.pdf'); path.parent.mkdir(exist_ok=True)
    styles=getSampleStyleSheet(); doc=SimpleDocTemplate(str(path),pagesize=A4)
    story=[Paragraph('Gemini Professional Data Analyst Report',styles['Title']),Spacer(1,12),Paragraph(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns",styles['Normal']),Spacer(1,12)]
    story.append(Paragraph('Data Quality',styles['Heading2']))
    q=quality['summary']; t=Table([['Metric','Value'],['Rows',f"{q['rows']:,}"],['Columns',q['columns']],['Missing cells',q['missing_cells']],['Duplicate rows',q['duplicate_rows']],['Quality score',q['score']]])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),.5,colors.grey),('PADDING',(0,0),(-1,-1),5)])); story += [t,Spacer(1,12),Paragraph('KPIs',styles['Heading2'])]
    for k,v in kpis.items(): story.append(Paragraph(f'<b>{k}</b>: {v}',styles['BodyText']))
    story += [Spacer(1,12),Paragraph('Executive Analysis',styles['Heading2'])]
    for line in executive.splitlines():
        if not line.strip():continue
        if line.startswith('#'): story.append(Paragraph(line.lstrip('# ').strip(),styles['Heading2']))
        else: story.append(Paragraph(line.replace('**',''),styles['BodyText']))
        story.append(Spacer(1,4))
    doc.build(story); return str(path)
