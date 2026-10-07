"""Financial analytics: KPIs, monthly/category/merchant/behaviour analysis, recurring payments,
unusual transactions, forecasting, budgets, goals, health score and rule-based insights.
Transfers and refunds are excluded from income and expense so they cannot inflate either."""
import math
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

if __package__:
    from .config import EXCLUDED
else:
    from config import EXCLUDED


def inr(x, sym="₹", d=0):
    """Indian digit grouping: 1234567 -> ₹12,34,567."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    neg, x = x < 0, abs(round(float(x), d))
    ip, _, fp = f"{x:.{d}f}".partition(".")
    if len(ip) > 3:
        ip = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", ip[:-3]) + "," + ip[-3:]
    return ("-" if neg else "") + sym + ip + ("." + fp if fp else "")


def month_label(m):
    return pd.Period(m, "M").strftime("%B %Y")


def prep(df):
    d = df.copy()
    d["transaction_date"] = pd.to_datetime(d["transaction_date"])
    d["month"] = d.transaction_date.dt.strftime("%Y-%m")
    d["weekday"] = d.transaction_date.dt.dayofweek
    d["is_weekend"] = d.weekday >= 5
    return d


def expenses(d):
    return d[(d.transaction_type == "Expense") & ~d.category.isin(EXCLUDED)]


def incomes(d):
    return d[(d.transaction_type == "Income") & ~d.category.isin(EXCLUDED)]


# ------------------------------------------------------------------ core aggregates
def kpis(d):
    e, i = expenses(d), incomes(d)
    inc, exp = float(i.amount.sum()), float(e.amount.sum())
    out = dict(income=inc, expense=exp, savings=inc - exp, savings_rate=(inc - exp) / inc if inc > 0 else None,
               n_txn=len(d), n_expense=len(e), avg_expense=float(e.amount.mean()) if len(e) else 0.0,
               median_expense=float(e.amount.median()) if len(e) else 0.0,
               max_expense=float(e.amount.max()) if len(e) else 0.0, min_expense=float(e.amount.min()) if len(e) else 0.0,
               largest=None, top_category=None, top_merchant=None)
    if len(e):
        r = e.loc[e.amount.idxmax()]
        out["largest"] = dict(date=r.transaction_date.date(), merchant=r.merchant, amount=float(r.amount), category=r.category)
        out["top_category"] = e.groupby("category").amount.sum().idxmax()
        out["top_merchant"] = e.groupby("merchant").amount.sum().idxmax()
    return out


def monthly(d):
    e = expenses(d).groupby("month").amount.sum()
    i = incomes(d).groupby("month").amount.sum()
    m = pd.DataFrame({"income": i, "expense": e}).fillna(0.0).sort_index()
    m["savings"] = m.income - m.expense
    m["savings_rate"] = np.where(m.income > 0, m.savings / m.income.replace(0, np.nan), np.nan)
    m["expense_change_pct"] = m.expense.pct_change() * 100
    return m


def category_table(d, month=None):
    e = expenses(d)
    cur = e[e.month == month] if month else e
    if cur.empty:
        return pd.DataFrame(columns=["category", "total", "count", "avg", "pct"])
    t = cur.groupby("category").amount.agg(total="sum", count="count", avg="mean").reset_index()
    t["pct"] = t.total / t.total.sum() * 100
    if month:
        prev = e[e.month == str(pd.Period(month, "M") - 1)].groupby("category").amount.sum()
        t["prev_total"] = t.category.map(prev).fillna(0.0)
        t["mom_growth_pct"] = np.where(t.prev_total > 0, (t.total / t.prev_total.replace(0, np.nan) - 1) * 100, np.nan)
    return t.sort_values("total", ascending=False).reset_index(drop=True)


def merchant_table(d, recurring_merchants=()):
    e = expenses(d)
    if e.empty:
        return pd.DataFrame(columns=["merchant", "total", "count", "avg", "first", "last", "recurring"])
    t = e.groupby("merchant").agg(total=("amount", "sum"), count=("amount", "count"), avg=("amount", "mean"),
                                  first=("transaction_date", "min"), last=("transaction_date", "max")).reset_index()
    t["first"], t["last"] = t["first"].dt.date, t["last"].dt.date
    t["recurring"] = t.merchant.isin(set(recurring_merchants))
    return t.sort_values("total", ascending=False).reset_index(drop=True)


def behavior(d):
    """Weekday vs weekend uses *per-day averages* (2 weekend days vs 5 weekdays are not comparable as totals)."""
    e = expenses(d)
    if e.empty:
        return {}
    days = pd.date_range(e.transaction_date.min().normalize(), e.transaction_date.max().normalize())
    n_wd, n_we = int((days.dayofweek < 5).sum()), int((days.dayofweek >= 5).sum())
    wd, we = float(e[~e.is_weekend].amount.sum()), float(e[e.is_weekend].amount.sum())
    wd_avg, we_avg = wd / max(n_wd, 1), we / max(n_we, 1)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    tot = e.groupby("weekday").amount.sum().reindex(range(7), fill_value=0.0)
    occ = pd.Series(days.dayofweek).value_counts().reindex(range(7), fill_value=0).replace(0, np.nan)
    dow = pd.DataFrame({"day": names, "total": tot.values, "avg_per_day": (tot / occ).fillna(0).values})
    dom = e.groupby(pd.cut(e.transaction_date.dt.day, [0, 5, 10, 15, 20, 25, 31],
                           labels=["1-5", "6-10", "11-15", "16-20", "21-25", "26-31"]), observed=False).amount.sum()
    has_time = bool((e.transaction_date.dt.hour != 0).any() or (e.transaction_date.dt.minute != 0).any())
    tod = None
    if has_time:  # most UPI statements carry no time; only analyse it when present
        tod = e.groupby(pd.cut(e.transaction_date.dt.hour, [-1, 5, 11, 16, 20, 23],
                               labels=["Night", "Morning", "Afternoon", "Evening", "Late evening"]), observed=False).amount.sum()
    return dict(weekday_total=wd, weekend_total=we, weekday_avg=wd_avg, weekend_avg=we_avg,
                weekend_vs_weekday_pct=(we_avg / wd_avg - 1) * 100 if wd_avg > 0 else None, dow=dow, dom=dom, tod=tod)


# ------------------------------------------------------------------ recurring payments
FREQ = [("Weekly", (5, 9), 52), ("Monthly", (25, 36), 12), ("Quarterly", (80, 100), 4), ("Yearly", (350, 380), 1)]


def detect_recurring(d):
    cols = ["merchant", "amount", "frequency", "confidence", "last_payment", "next_estimated_payment", "annual_cost", "occurrences"]
    rows = []
    e = expenses(d)
    if "subcategory" in e:
        e = e[e.subcategory != "Cash Withdrawal"]   # ATM withdrawals are not subscriptions
    for m, g in e.groupby("merchant"):
        g = g.sort_values("transaction_date")
        if len(g) < 3:
            continue
        a = g.amount.values
        med = float(np.median(a))
        dev = np.abs(a - med) / med if med > 0 else np.ones(len(a))
        if (dev > 0.15).mean() > 0.2:
            continue
        days = np.diff(g.transaction_date.values.astype("datetime64[D]")).astype(int)
        md = float(np.median(days))
        for name, (lo, hi), per in FREQ:
            if lo <= md <= hi:
                regular = float(((days >= lo) & (days <= hi)).mean())
                if regular < 0.6:
                    break
                conf = round(0.4 * regular + 0.3 * float(1 - dev.mean()) + 0.3 * min(len(g) / 6, 1), 2)
                if conf >= 0.6:
                    last = g.transaction_date.max()
                    rows.append([m, med, name, conf, last.date(), (last + pd.Timedelta(days=int(round(md)))).date(), med * per, len(g)])
                break
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("annual_cost", ascending=False).reset_index(drop=True) if len(out) else out


# ------------------------------------------------------------------ unusual transactions
def detect_anomalies(d):
    """Flags 'unusual' (never 'fraud') expenses: robust per-category rule + Isolation Forest, with human-readable reasons."""
    e = expenses(d).copy()
    e["is_anomaly"], e["anomaly_reason"] = False, ""
    if len(e) >= 10:
        # Baseline = the merchant's own history when it has >=6 payments, otherwise its category/subcategory.
        mc = e.groupby("merchant").amount.transform("count")
        key = e.merchant.where(mc >= 6, e.category + " / " + e.subcategory)
        label = e.merchant.where(mc >= 6, e.category)
        g = e.groupby(key).amount
        med, cnt = g.transform("median"), g.transform("count")
        mad = g.transform(lambda x: np.median(np.abs(x - np.median(x))))
        ratio = e.amount / med.replace(0, np.nan)
        rule = (cnt >= 6) & (ratio >= 3) & (e.amount > med + 3.5 * 1.4826 * mad.clip(lower=1))
        iso = pd.Series(False, index=e.index)
        if len(e) >= 30:
            X = np.column_stack([np.log1p(e.amount), np.log1p(ratio.fillna(1)), np.log1p(mc), e.weekday])
            iso = pd.Series(IsolationForest(contamination=0.03, random_state=42).fit_predict(X) == -1, index=e.index)
        flag = rule | (iso & (ratio >= 2.5) & (cnt >= 6))
        e["is_anomaly"] = flag
        reason = ("Amount " + e.amount.map(inr) + " is about " + ratio.round(1).astype(str) + "x your typical "
                  + label + " transaction (" + med.map(inr) + ").")
        e.loc[flag, "anomaly_reason"] = reason[flag]
    if "user_status" in e:
        e.loc[e.user_status == "normal", ["is_anomaly", "anomaly_reason"]] = [False, ""]
        e.loc[e.user_status == "suspicious", "is_anomaly"] = True
        e.loc[(e.user_status == "suspicious") & (e.anomaly_reason == ""), "anomaly_reason"] = "Marked as suspicious by you."
    return e


def enrich(df):
    d = prep(df)
    a = detect_anomalies(d)
    d["is_anomaly"] = a["is_anomaly"].reindex(d.index).fillna(False).astype(bool)
    d["anomaly_reason"] = a["anomaly_reason"].reindex(d.index).fillna("")
    rec = detect_recurring(d)
    d["is_recurring"] = d.merchant.isin(set(rec.merchant)) & (d.transaction_type == "Expense")
    return d


# ------------------------------------------------------------------ forecast
def forecast(d, drop_partial=True):
    """Estimated (never guaranteed) next-month expense. Monthly series are short, so only simple, testable models are used."""
    e = expenses(d)
    if e.empty:
        return dict(ok=False, msg="No expense data yet.")
    s = e.groupby("month").amount.sum().sort_index()
    last, dropped = d.transaction_date.max(), None
    if drop_partial and len(s) > 1 and (last + pd.offsets.MonthEnd(0) - last).days > 3 and s.index[-1] == last.strftime("%Y-%m"):
        dropped, s = s.index[-1], s.iloc[:-1]
    if len(s) < 3:
        return dict(ok=False, msg=f"At least 3 complete months of expenses are needed for a forecast (found {len(s)}).")
    y, n = s.values.astype(float), len(s)
    ma = lambda h: float(h[-3:].mean())
    lin = lambda h: float(LinearRegression().fit(np.arange(len(h)).reshape(-1, 1), h).predict([[len(h)]])[0])
    methods = {"3-month moving average": ma, "Linear trend": lin,
               "Average of moving average and linear trend": lambda h: (ma(h) + lin(h)) / 2}
    errs = {k: [y[t] - f(y[:t]) for t in range(3, n)] for k, f in methods.items()}
    if n >= 6:
        best = min(list(methods)[:2], key=lambda k: np.mean(np.abs(errs[k])))
    else:
        best = "Average of moving average and linear trend"
    pred = max(methods[best](y), 0.0)
    e_best = np.array(errs[best])
    sd = float(np.std(e_best)) if len(e_best) >= 2 else float(np.std(np.diff(y)))
    sd = max(sd, 0.03 * pred)
    pv = e.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0).loc[s.index]
    rising = None
    if n >= 4:
        chg = (pv.iloc[-1] - pv.iloc[-4:-1].mean()).sort_values(ascending=False)
        if chg.iloc[0] > 0:
            rising = (chg.index[0], float(chg.iloc[0]))
    slope = float(LinearRegression().fit(np.arange(n).reshape(-1, 1), y).coef_[0])
    return dict(ok=True, predicted=pred, low=max(pred - 1.28 * sd, 0.0), high=pred + 1.28 * sd, method=best, n_months=n,
                mae=float(np.mean(np.abs(e_best))) if len(e_best) else None,
                mape=float(np.mean(np.abs(e_best) / y[3:n][:len(e_best)]) * 100) if len(e_best) else None,
                history=s, next_month=str(pd.Period(s.index[-1], "M") + 1), slope=slope, rising=rising, dropped_partial=dropped,
                explanation=f"Estimate based on the last {n} complete months using {best.lower()}; the range is an approximate 80% band from past forecast errors.")


# ------------------------------------------------------------------ budgets, goals, health
def budget_status(d, budgets, month):
    spent = expenses(d)[lambda x: x.month == month].groupby("category").amount.sum()
    rows = []
    for cat, lim in budgets.items():
        a = float(spent.get(cat, 0.0))
        u = a / lim * 100 if lim > 0 else 0
        rows.append(dict(category=cat, limit=lim, actual=a, remaining=lim - a, utilization_pct=u,
                         status="🔴 EXCEEDED" if u > 100 else "🟡 NEAR LIMIT" if u >= 80 else "🟢 SAFE"))
    return pd.DataFrame(rows, columns=["category", "limit", "actual", "remaining", "utilization_pct", "status"])


def avg_monthly_savings(d, n=3):
    m = monthly(d)
    return float(m.savings.tail(n).mean()) if len(m) else 0.0


def goal_progress(g, avg_savings):
    target, cur = float(g["target_amount"]), float(g["current_amount"])
    rem = max(target - cur, 0.0)
    contrib = float(g["monthly_contribution"] or 0) or max(avg_savings, 0.0)
    basis = "your planned monthly contribution" if float(g["monthly_contribution"] or 0) > 0 else "your recent average savings"
    months = math.ceil(rem / contrib) if contrib > 0 and rem > 0 else (0 if rem == 0 else None)
    est = (pd.Timestamp.today().normalize() + pd.DateOffset(months=months)).date() if months is not None else None
    need = None
    if g.get("target_date"):
        td = pd.to_datetime(g["target_date"], errors="coerce")
        if pd.notna(td) and td > pd.Timestamp.today():
            need = rem / max((td.year - pd.Timestamp.today().year) * 12 + td.month - pd.Timestamp.today().month, 1)
    return dict(progress=min(cur / target * 100, 100) if target > 0 else 0, remaining=rem, est_months=months,
                est_date=est, basis=basis, required_monthly=need)


def concentration(d, k=3):
    e = expenses(d)
    if e.empty:
        return None
    s = e.groupby("category").amount.sum().sort_values(ascending=False)
    return dict(top=list(s.index[:k]), amount=float(s.iloc[:k].sum()), share=float(s.iloc[:k].sum() / s.sum() * 100))


def health_score(d, budgets, rec):
    """Transparent 100-point analytics score (not financial advice)."""
    m = monthly(d)
    if m.empty:
        return None
    comp, last = {}, m.iloc[-1]
    sr = last.savings_rate if pd.notna(last.savings_rate) else 0
    comp["Savings rate (35)"] = round(35 * min(max(sr, 0) / 0.30, 1), 1)
    if len(m) >= 4 and m.expense.iloc[-4:-1].mean() > 0:
        g = m.expense.iloc[-1] / m.expense.iloc[-4:-1].mean() - 1
        comp["Expense growth (20)"] = round(20 * (1 - min(max(g, 0) / 0.25, 1)), 1)
    else:
        comp["Expense growth (20)"] = 10.0
    if budgets:
        b = budget_status(d, budgets, m.index[-1])
        comp["Budget adherence (20)"] = round(20 * float((b.actual <= b.limit).mean()), 1)
    else:
        comp["Budget adherence (20)"] = 10.0
    c = concentration(d)
    comp["Spending concentration (10)"] = round(10 * (1 - min(max((c["share"] - 50) / 35, 0), 1)), 1) if c else 5.0
    monthly_rec = float(rec[rec.frequency == "Monthly"].amount.sum()) if len(rec) else 0.0
    burden = monthly_rec / last.income if last.income > 0 else 0.4
    comp["Recurring burden (15)"] = round(15 * (1 - min(max((burden - 0.10) / 0.30, 0), 1)), 1)
    score = round(sum(comp.values()))
    label = "Excellent" if score >= 80 else "Good" if score >= 65 else "Fair" if score >= 50 else "Needs attention"
    return dict(score=score, label=label, components=comp, no_budgets=not budgets)


# ------------------------------------------------------------------ narrative pieces
def month_over_month(d, month):
    e = expenses(d)
    prev = str(pd.Period(month, "M") - 1)
    cur_t, prev_t = float(e[e.month == month].amount.sum()), float(e[e.month == prev].amount.sum())
    if prev_t <= 0:
        return None
    ct = category_table(d, month)
    drivers = ct.assign(diff=ct.total - ct.prev_total).sort_values("diff", ascending=False)
    up = [r.category for r in drivers.itertuples() if r.diff > 0][:2] if cur_t > prev_t else \
         [r.category for r in drivers.sort_values("diff").itertuples() if r.diff < 0][:2]
    word = "increased" if cur_t > prev_t else "decreased"
    expl = (f"Your {month_label(month)} expenses {word} by {inr(abs(cur_t - prev_t))} ({abs(cur_t / prev_t - 1) * 100:.1f}%)"
            + (f", mainly because {' and '.join(up)} spending {word}." if up else "."))
    return dict(prev_month=prev, current=cur_t, previous=prev_t, change=cur_t - prev_t, change_pct=(cur_t / prev_t - 1) * 100, explanation=expl)


def make_insights(k, cats, mom, beh, rec, conc):
    out = []
    if k["top_category"] and len(cats):
        out.append(f"{cats.iloc[0].category} is your highest spending category ({inr(cats.iloc[0].total)}, {cats.iloc[0].pct:.1f}% of expenses).")
    if mom:
        out.append(f"Your expenses {'increased' if mom['change'] > 0 else 'decreased'} {abs(mom['change_pct']):.0f}% compared with {month_label(mom['prev_month'])}.")
    fd = cats[cats.category == "Food"]
    if len(fd):
        out.append(f"You spent {inr(fd.iloc[0].total)} on Food.")
    if beh and beh.get("weekend_vs_weekday_pct") is not None:
        p = beh["weekend_vs_weekday_pct"]
        out.append(f"Per day, you spend {abs(p):.0f}% {'more' if p > 0 else 'less'} on weekends than on weekdays "
                   f"({inr(beh['weekend_avg'])} vs {inr(beh['weekday_avg'])} per day).")
    if len(rec):
        out.append(f"{len(rec)} possible recurring payment(s) detected, about {inr(float(rec[rec.frequency == 'Monthly'].amount.sum()))} per month.")
    if k["savings_rate"] is not None:
        out.append(f"Your savings rate is {k['savings_rate'] * 100:.1f}% ({inr(k['savings'])} retained after recorded expenses).")
    if conc:
        out.append(f"About {conc['share']:.0f}% of your spending comes from your top three categories ({', '.join(conc['top'])}).")
    return out


def make_recommendations(k, cats):
    out = []
    if len(cats):
        t = cats.iloc[0]
        out.append(f"You spent {inr(t.total)} on {t.category} ({t.pct:.0f}% of expenses). Reducing it by 10% could free up roughly {inr(t.total * 0.10)} a month.")
    if k["savings_rate"] is not None and k["savings_rate"] < 0.20:
        out.append("Your savings rate is below 20%. Setting category budgets for your top two categories is a practical first step.")
    return out


def build_bundle(d, month, budgets):
    """Everything the UI, reports and AI need, computed once from verified data."""
    months = sorted(d.month.unique())
    fm = month or months[-1]
    dm = d[d.month == month] if month else d
    rec = detect_recurring(d)
    k = kpis(dm)
    cats = category_table(d, month) if month else category_table(d)
    mom = month_over_month(d, fm)
    beh = behavior(dm)
    conc = concentration(dm)
    fc = forecast(d)
    bud = budget_status(d, budgets, fm) if budgets else pd.DataFrame()
    return dict(month=month, focus_month=fm, dm=dm, kpis=k, monthly=monthly(d), cats=cats, merchants=merchant_table(dm, rec.merchant),
                behavior=beh, recurring=rec, anomalies=dm[dm.is_anomaly & (dm.transaction_type == "Expense")], mom=mom,
                concentration=conc, forecast=fc, budget=bud, health=health_score(d, budgets, rec),
                insights=make_insights(k, cats, mom, beh, rec, conc), recs=make_recommendations(k, cats),
                avg_savings=avg_monthly_savings(d), months=months)
