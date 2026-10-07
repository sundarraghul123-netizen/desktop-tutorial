"""Generate a realistic, messy UPI statement (Feb-Sep 2026) for demos and tests."""
import numpy as np, pandas as pd
from pathlib import Path

def generate(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    ref = lambda: str(rng.integers(10**11, 10**12))
    def add(date, desc, amt, typ): rows.append([date.strftime("%d-%m-%Y"), desc, f"{amt:.2f}", typ, ref()])
    styles = ["UPI/{r}/{m}", "UPI-{m}-MUMBAI", "PAYTM*{m}", "{m} ONLINE", "{m} PVT LTD"]
    def merch(date, name, lo, hi, n):
        for _ in range(n):
            day = int(rng.integers(1, date.days_in_month + 1))
            d = date.replace(day=day)
            add(d, styles[rng.integers(0, 5)].format(r=ref()[:6], m=name), round(rng.uniform(lo, hi), 0), "Debit")
    for m in range(2, 10):
        mo = pd.Timestamp(2026, m, 1)
        add(mo, f"NEFT-ACME TECHNOLOGIES PVT LTD-SALARY {mo.strftime('%b').upper()}", 75000, "Credit")
        add(mo.replace(day=5), "UPI-RAVI KUMAR-RENT-9876543@ybl", 18000, "Debit")
        add(mo.replace(day=5), "UPI-NETFLIX-NETFLIX.COM", 649, "Debit")
        add(mo.replace(day=12), "UPI-SPOTIFY INDIA-SPOTIFY", 119, "Debit")
        add(mo.replace(day=8), "UPI-ACT FIBERNET-BROADBAND", 999, "Debit")
        add(mo.replace(day=15), "BESCOM ELECTRICITY BILL", round(rng.uniform(900, 1700)), "Debit")
        add(mo.replace(day=3), "UPI-CULT FIT-MEMBERSHIP", 1500, "Debit")
        scale = 1.3 if m == 9 else 1.0   # September spending rises
        merch(mo, "SWIGGY", 150, 700 * scale, int(12 * scale)); merch(mo, "ZOMATO", 150, 650, 4)
        merch(mo, "BLINKIT", 300, 1500, 4); merch(mo, "BIGBASKET", 500, 2200, 2)
        merch(mo, "STARBUCKS", 250, 600, 3); merch(mo, "UBER", 80, 450, 6); merch(mo, "OLA", 80, 380, 3)
        merch(mo, "RAPIDO", 40, 200, 3); merch(mo, "INDIAN OIL", 800, 1500, 2)
        merch(mo, "AMAZON", 300, 3500 * scale, 3); merch(mo, "MYNTRA", 800, 2500, 1)
        merch(mo, "ABC STORE", 200, 1250, 2)
        for _ in range(3):
            add(mo.replace(day=int(rng.integers(1, 28))), f"UPI-RAHUL SHARMA-{ref()[:6]}@okaxis", round(rng.uniform(200, 1500)), "Debit")
        add(mo.replace(day=20), "UPI-SELF TRANSFER-OWN ACCOUNT", 5000, "Debit"); add(mo.replace(day=20), "UPI-SELF TRANSFER-OWN ACCOUNT", 5000, "Credit")
        add(mo.replace(day=25), "ATM CASH WITHDRAWAL", 2000, "Debit")
    add(pd.Timestamp(2026, 9, 18), "AMAZON PAY INDIA PVT LTD", 18500, "Debit")        # unusual transaction
    add(pd.Timestamp(2026, 7, 22), "AMAZON REFUND", 899, "Credit")
    add(pd.Timestamp(2026, 9, 30), "INT.PD:SAVINGS A/C", 312.5, "Credit")
    df = pd.DataFrame(rows, columns=["Date", "Description", "Amount", "Type", "Ref No"])
    df = df.iloc[np.argsort(pd.to_datetime(df.Date, dayfirst=True).values, kind="stable")].reset_index(drop=True)
    dups = df.sample(12, random_state=1)                      # re-exported duplicate rows (same Ref No)
    bad = pd.DataFrame([["31-02-2026", "BAD DATE ROW", "100", "Debit", ""], ["05-03-2026", "BAD AMOUNT ROW", "abc", "Debit", ""]], columns=df.columns)
    return pd.concat([df, dups, bad], ignore_index=True)

if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "sample" / "sample_upi_statement.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(out, index=False); print("wrote", out)
