"""Monthly report generation (Excel + PDF). PDF uses 'Rs.' because the built-in PDF fonts have no rupee glyph."""
import io
from datetime import date
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

if __package__:
    from .analytics import inr, month_label
else:
    from analytics import inr, month_label

DISCLAIMER = ("This report is an analytics and educational summary. Forecasts are estimates, unusual transactions are not "
              "confirmed fraud, and nothing here is regulated financial advice.")


def report_name(month, ext):
    return f"UPI_Financial_Report_{month_label(month).replace(' ', '_')}.{ext}" if month else f"UPI_Financial_Report_All_Time.{ext}"


def _safe(v):
    """Neutralise spreadsheet formula injection (cells starting with = + - @)."""
    return "'" + v if isinstance(v, str) and v and v[0] in "=+-@" else v


def _sheet(df):
    if df is None or len(df.columns) == 0:
        return pd.DataFrame({"info": ["No data"]})
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].map(_safe)
    return df if len(df) else pd.concat([df, pd.DataFrame([{df.columns[0]: "No data"}])], ignore_index=True)


def excel_report(b):
    k, fc = b["kpis"], b["forecast"]
    summary = pd.DataFrame([
        ("Period", month_label(b["month"]) if b["month"] else "All time"), ("Income", k["income"]), ("Expenses", k["expense"]),
        ("Savings", k["savings"]), ("Savings rate", "n/a" if k["savings_rate"] is None else f"{k['savings_rate'] * 100:.1f}%"),
        ("Transactions", k["n_txn"]), ("Top category", k["top_category"] or "-"), ("Top merchant", k["top_merchant"] or "-"),
        ("Health score", f"{b['health']['score']}/100 ({b['health']['label']})" if b["health"] else "-"),
        ("Note", "Excludes transfers and refunds. Analytics only, not financial advice.")], columns=["Metric", "Value"])
    tx = b["dm"][["transaction_date", "description", "merchant", "amount", "transaction_type", "category", "subcategory",
                  "category_confidence", "is_recurring", "is_anomaly"]].copy()
    tx["transaction_date"] = tx.transaction_date.dt.date
    fdf = pd.DataFrame([("Estimated next-month expense", fc["predicted"]), ("Range low", fc["low"]), ("Range high", fc["high"]),
                        ("Method", fc["method"]), ("Months used", fc["n_months"])], columns=["Item", "Value"]) if fc["ok"] else pd.DataFrame({"info": [fc["msg"]]})
    sheets = {"Summary": summary, "Transactions": tx, "Categories": b["cats"], "Merchants": b["merchants"],
              "Monthly Analysis": b["monthly"].reset_index().rename(columns={"index": "month"}), "Recurring Payments": b["recurring"],
              "Anomalies": b["anomalies"][["transaction_date", "merchant", "amount", "category", "anomaly_reason"]].assign(
                  transaction_date=lambda x: x.transaction_date.dt.date), "Budget": b["budget"], "Forecast": fdf}
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df = _sheet(df)
            df.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            for i, col in enumerate(df.columns, 1):
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(len(str(col)), *(len(str(x)) for x in df[col].head(50))) + 2, 60)
    return buf.getvalue()


def _tbl(rows, header, widths=None):
    t = Table([header] + rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")])]))
    return t


def _vbar(labels, values, w=480, h=150):
    d = Drawing(w, h); c = VerticalBarChart()
    c.x, c.y, c.width, c.height = 45, 25, w - 60, h - 40
    c.data, c.categoryAxis.categoryNames = [values], labels
    c.bars[0].fillColor = colors.HexColor("#2563eb"); c.valueAxis.valueMin = 0
    c.categoryAxis.labels.fontSize = c.valueAxis.labels.fontSize = 7
    d.add(c); return d


def _hbar(labels, values, w=480, h=150):
    d = Drawing(w, h); c = HorizontalBarChart()
    c.x, c.y, c.width, c.height = 90, 15, w - 110, h - 25
    c.data, c.categoryAxis.categoryNames = [values[::-1]], labels[::-1]
    c.bars[0].fillColor = colors.HexColor("#0d9488"); c.valueAxis.valueMin = 0
    c.categoryAxis.labels.fontSize = c.valueAxis.labels.fontSize = 7
    d.add(c); return d


def pdf_report(b):
    rs = lambda x: inr(x, "Rs. ")
    st = getSampleStyleSheet(); H, P = st["Heading2"], st["BodyText"]
    k, fc, mom = b["kpis"], b["forecast"], b["mom"]
    title = month_label(b["month"]) if b["month"] else "All time"
    s = [Paragraph("UPI FINANCIAL REPORT", st["Title"]), Paragraph(escape(title), st["Heading3"]),
         Paragraph(f"Generated {date.today():%d %b %Y}", P), Spacer(1, 6)]
    s += [Paragraph("1. Executive Summary", H), _tbl([
        [rs(k["income"]), rs(k["expense"]), rs(k["savings"]), "n/a" if k["savings_rate"] is None else f"{k['savings_rate'] * 100:.1f}%"]],
        ["Income", "Expenses", "Savings", "Savings rate"]),
        Paragraph(f"Top category: {escape(str(k['top_category']))} &nbsp; | &nbsp; Top merchant: {escape(str(k['top_merchant']))} &nbsp; | &nbsp; "
                  f"Transactions: {k['n_txn']}" + (f" &nbsp; | &nbsp; Health score: {b['health']['score']}/100 ({b['health']['label']})" if b["health"] else ""), P)]
    m = b["monthly"].tail(12)
    s += [Paragraph("2-4. Income, Expense and Savings by Month", H), _vbar([month_label(i)[:3] + " " + i[2:4] for i in m.index], [float(x) for x in m.expense]),
          _tbl([[month_label(i), rs(r.income), rs(r.expense), rs(r.savings)] for i, r in m.iterrows()], ["Month", "Income", "Expense", "Savings"])]
    c = b["cats"].head(8)
    if len(c):
        s += [Paragraph("5. Category Analysis", H), _hbar(list(c.category), [float(x) for x in c.total]),
              _tbl([[r.category, rs(r.total), f"{r.pct:.1f}%", int(r.count), rs(r.avg)] for r in c.itertuples()], ["Category", "Total", "Share", "Count", "Average"])]
    mt = b["merchants"].head(8)
    if len(mt):
        s += [Paragraph("6. Merchant Analysis", H), _tbl([[escape(r.merchant), rs(r.total), int(r.count), rs(r.avg), "Yes" if r.recurring else ""] for r in mt.itertuples()],
                                                         ["Merchant", "Total", "Count", "Average", "Recurring"])]
    s += [Paragraph("7. Month-over-Month Comparison", H), Paragraph(escape(mom["explanation"]) if mom else "No previous month available.", P)]
    bh = b["behavior"]
    if bh:
        s += [Paragraph("8. Spending Behaviour", H), Paragraph(
            f"Average per weekday: {rs(bh['weekday_avg'])}; per weekend day: {rs(bh['weekend_avg'])}"
            + (f" ({bh['weekend_vs_weekday_pct']:+.0f}%)." if bh["weekend_vs_weekday_pct"] is not None else "."), P)]
    r = b["recurring"]
    s += [Paragraph("9. Recurring Payments", H)] + ([_tbl([[escape(x.merchant), rs(x.amount), x.frequency, str(x.next_estimated_payment), rs(x.annual_cost)] for x in r.itertuples()],
                                                         ["Merchant", "Amount", "Frequency", "Next estimated", "Annual cost"])] if len(r) else [Paragraph("None detected.", P)])
    a = b["anomalies"]
    s += [Paragraph("10. Unusual Transactions (not confirmed fraud)", H)] + ([_tbl([[str(x.transaction_date.date()), escape(x.merchant), rs(x.amount), escape(x.anomaly_reason[:70])] for x in a.head(8).itertuples()],
                                                                                  ["Date", "Merchant", "Amount", "Reason"])] if len(a) else [Paragraph("None detected.", P)])
    bd = b["budget"]
    s += [Paragraph("11. Budget Performance", H)] + ([_tbl([[x.category, rs(x.limit), rs(x.actual), f"{x.utilization_pct:.0f}%", x.status[2:]] for x in bd.itertuples()],
                                                          ["Category", "Budget", "Actual", "Used", "Status"])] if len(bd) else [Paragraph("No budgets set.", P)])
    s += [Paragraph("12. Forecast (estimate)", H), Paragraph(
        (f"Estimated next-month expense: {rs(fc['predicted'])} (likely range {rs(fc['low'])} to {rs(fc['high'])}). {escape(fc['explanation'])}") if fc["ok"] else escape(fc["msg"]), P)]
    s += [Paragraph("13. Insights", H)] + [Paragraph("&bull; " + escape(i), P) for i in b["insights"]]
    s += [Paragraph("14. Recommendations", H)] + [Paragraph("&bull; " + escape(i), P) for i in b["recs"]] + [Spacer(1, 10), Paragraph(f"<i>{DISCLAIMER}</i>", P)]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                      title="UPI Financial Report").build(s)
    return buf.getvalue()
