"""File reading, column detection, cleaning, validation and duplicate hashing."""
import hashlib
import io
import re

import numpy as np
import pandas as pd

if __package__:
    from .config import MAX_UPLOAD_MB
else:
    from config import MAX_UPLOAD_MB

CANON = ["date", "description", "amount", "type", "debit", "credit", "ref", "balance"]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


ALIASES = {
    "date": ["date", "transactiondate", "txndate", "valuedate", "postingdate", "trandate", "bookingdate", "paymentdate"],
    "description": ["description", "narration", "transactiondetails", "remarks", "particulars", "details",
                    "transactionremarks", "transactiondescription"],
    "amount": ["amount", "transactionamount", "amt"],
    "type": ["type", "transactiontype", "drcr", "crdr", "txntype", "debitcredit"],
    "debit": ["debit", "withdrawal", "withdrawalamt", "withdrawalamount", "debitamount", "dr", "debits", "moneyout", "paidout"],
    "credit": ["credit", "deposit", "depositamt", "depositamount", "creditamount", "cr", "credits", "moneyin"],
    "ref": ["reference", "refno", "referenceno", "chqrefno", "utr", "utrno", "transactionid", "txnid", "upiref", "upitransactionid"],
    "balance": ["balance", "closingbalance", "runningbalance"],
}


def _find_header(raw):
    """Bank exports often have junk rows above the real header; pick the row matching most aliases."""
    best, idx = 0, 0
    for i in range(min(30, len(raw))):
        vals = {_norm(v) for v in raw.iloc[i].tolist() if pd.notna(v)}
        score = sum(any(v in al for v in vals) for al in ALIASES.values())
        if score > best:
            best, idx = score, i
    return idx if best >= 2 else 0


def _pdf_date(text):
    text = re.sub(r"(?i)\b(?:date|time)\b", " ", text)
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce", format="mixed")
    return parsed if not pd.isna(parsed) else None


def _read_pdf(data, progress_callback=None):
    """OCR image-based statement PDFs locally and return transaction-shaped columns."""
    try:
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise ValueError(
            "PDF OCR support is unavailable. Install the project requirements with "
            "`python -m pip install -r requirements.txt`."
        ) from e

    try:
        document = pdfium.PdfDocument(data)
    except Exception as e:
        raise ValueError("Could not open this PDF. Please check that it is a valid statement.") from e

    transactions = []
    anchor_re = re.compile(r"\b(paid\s+to|received\s+from)\b", re.IGNORECASE)
    try:
        if len(document) > 200:
            raise ValueError("This PDF has too many pages to process (maximum 200).")
        ocr = RapidOCR()
        for page_index in range(len(document)):
            page = document[page_index]
            bitmap = page.render(scale=2.0)
            image = bitmap.to_numpy()
            del bitmap
            top_fraction = 0.38 if page_index == 0 else 0.04
            image = image[int(image.shape[0] * top_fraction):]
            try:
                result, _ = ocr(image)
            except Exception as e:
                raise ValueError(
                    f"Could not OCR PDF page {page_index + 1}. Please try a clearer PDF."
                ) from e
            finally:
                page.close()

            items = []
            for box, text, _score in result or []:
                text = str(text).strip()
                if not text:
                    continue
                x = min(point[0] for point in box)
                y = sum(point[1] for point in box) / len(box)
                items.append((y, x, text))

            anchors = [
                (y, x, text, match)
                for y, x, text in items
                if (match := anchor_re.search(text)) and 160 <= x <= 850
            ]
            anchors.sort(key=lambda item: item[0])
            for index, (y, x, text, match) in enumerate(anchors):
                next_y = anchors[index + 1][0] if index + 1 < len(anchors) else float("inf")
                band = [item for item in items if y - 12 <= item[0] < min(next_y, y + 70)]

                detail = [
                    item[2] for item in band
                    if 160 <= item[1] < 730 and abs(item[0] - y) <= 18
                ]
                description = " ".join(detail)
                description = anchor_re.sub("", description, count=1).strip(" :-")

                date_candidates = [
                    (abs(item[0] - y), parsed)
                    for item in items
                    if item[1] < 180 and abs(item[0] - y) <= 40
                    if (parsed := _pdf_date(item[2])) is not None
                ]
                date = min(date_candidates, default=(0, None), key=lambda item: item[0])[1]

                amount_candidates = []
                for item_y, item_x, amount_text in band:
                    if item_x < 900:
                        continue
                    match_amount = re.search(r"(?<!\w)(\d[\d,]*(?:\.\d{1,2})?)", amount_text)
                    if match_amount:
                        try:
                            amount = float(match_amount.group(1).replace(",", ""))
                        except ValueError:
                            continue
                        if amount > 0:
                            amount_candidates.append((abs(item_y - y), amount))
                amount = min(amount_candidates, default=(0, None), key=lambda item: item[0])[1]

                transactions.append({
                    "Date": date,
                    "Description": description or text,
                    "Amount": amount,
                    "Type": "Income" if match.group(1).lower().startswith("received") else "Expense",
                })
            if progress_callback is not None:
                progress_callback(page_index + 1, len(document))
    finally:
        document.close()

    if not transactions:
        raise ValueError(
            "No transactions could be recognized in this PDF. It may not be a supported "
            "statement layout or the scan may be too unclear."
        )
    return pd.DataFrame(transactions)


def read_file(name, data, progress_callback=None):
    """Validate size/extension and return a raw DataFrame (all strings). Raises ValueError with a user-friendly message."""
    if not data:
        raise ValueError("The file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_UPLOAD_MB} MB.")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "pdf":
        return _read_pdf(data, progress_callback=progress_callback)
    if ext not in {"csv", "xlsx", "xls"}:
        raise ValueError("Unsupported format. Please upload a PDF, CSV, XLSX or XLS file.")
    raw = None
    try:
        if ext == "csv":
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    raw = pd.read_csv(io.BytesIO(data), header=None, dtype=str, encoding=enc, sep=None,
                                      engine="python", on_bad_lines="skip")
                    break
                except UnicodeDecodeError:
                    continue
        else:
            raw = pd.read_excel(io.BytesIO(data), header=None, dtype=str)
    except ImportError as e:
        raise ValueError(f"Missing library to read this file type: {e}. Run: pip install xlrd openpyxl")
    except Exception as e:  # corrupted / malicious / unreadable
        raise ValueError(f"Could not read the file ({type(e).__name__}). Please check that it is a valid statement.")
    if raw is None or raw.dropna(how="all").empty:
        raise ValueError("The file is empty or could not be parsed.")
    raw = raw.dropna(how="all").reset_index(drop=True)
    h = _find_header(raw)
    cols, seen = [], {}
    for i, c in enumerate(raw.iloc[h].tolist()):
        c = str(c).strip() if pd.notna(c) else f"col_{i}"
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df = raw.iloc[h + 1:].reset_index(drop=True)
    df.columns = cols
    if df.empty:
        raise ValueError("No transaction rows found below the header.")
    return df


def detect_columns(df):
    mapping, used = {}, set()
    for canon in CANON:
        mapping[canon] = None
        for c in df.columns:
            if c not in used and _norm(c) in ALIASES[canon]:
                mapping[canon] = c
                used.add(c)
                break
    return mapping


def parse_amounts(s):
    """Return (signed float values, side hint from Dr/Cr suffix)."""
    s = s.astype(str).str.strip()
    side = pd.Series(np.where(s.str.contains(r"(?:dr|debit)\.?\s*$", case=False, regex=True), "Expense",
                              np.where(s.str.contains(r"(?:cr|credit)\.?\s*$", case=False, regex=True), "Income", "")),
                     index=s.index)
    neg = s.str.contains(r"^\s*[-(]", regex=True)
    num = pd.to_numeric(s.str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    return num.where(~neg, -num), side


def _clean_desc(x):
    x = re.sub(r"[\x00-\x1f\x7f<>]", " ", str(x))        # control chars + HTML-ish brackets (XSS / injection hygiene)
    return re.sub(r"\s+", " ", x).strip()[:200]


def payment_method(desc):
    d = desc.upper()
    for key, label in (("UPI", "UPI"), ("IMPS", "IMPS"), ("NEFT", "NEFT"), ("RTGS", "RTGS"), ("ATM", "ATM"),
                       ("POS", "Card"), ("ECOM", "Card"), ("NACH", "Auto-debit"), ("ACH", "Auto-debit")):
        if re.search(rf"\b{key}\b", d):
            return label
    return "Other"


def clean(df, m, dayfirst=True, positive_amount_type="Income"):
    """Return (ok, bad, stats). `ok` has: transaction_date, description, amount(+), transaction_type, ref, payment_method, txn_hash."""
    if positive_amount_type not in {"Income", "Expense"}:
        raise ValueError("Positive amounts must be classified as Income or Expense.")
    if not m.get("date"):
        raise ValueError("Unable to identify the transaction date column. Please select the correct date column.")
    if not m.get("description"):
        raise ValueError("Unable to identify the description column. Please select the correct description column.")
    if not (m.get("amount") or m.get("debit") or m.get("credit")):
        raise ValueError("Unable to identify the amount column(s). Please select Amount, or Debit/Credit columns.")

    d = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
    n_rows = len(d)
    out = pd.DataFrame(index=d.index)
    out["transaction_date"] = pd.to_datetime(d[m["date"]], dayfirst=dayfirst, errors="coerce", format="mixed")
    out["description"] = d[m["description"]].fillna("").map(_clean_desc)
    reason = pd.Series("", index=d.index)
    reason[out.transaction_date.isna()] = "invalid or missing date"
    reason[(reason == "") & (out.description == "")] = "missing description"

    amt_cols = [c for c in (m.get("amount"), m.get("debit"), m.get("credit")) if c]
    cur = d[amt_cols].astype(str).apply(lambda s: s.str.contains(r"[$€£]|\busd\b|\beur\b|\bgbp\b", case=False, regex=True)).any(axis=1)
    reason[(reason == "") & cur] = "unsupported currency"

    if m.get("debit") or m.get("credit"):
        deb = parse_amounts(d[m["debit"]])[0].abs().fillna(0) if m.get("debit") else pd.Series(0.0, index=d.index)
        cre = parse_amounts(d[m["credit"]])[0].abs().fillna(0) if m.get("credit") else pd.Series(0.0, index=d.index)
        out["amount"] = deb + cre
        out["transaction_type"] = np.where((deb > 0) & (cre == 0), "Expense", np.where((cre > 0) & (deb == 0), "Income", ""))
        reason[(reason == "") & ~((deb > 0) ^ (cre > 0))] = "invalid amount (need exactly one of debit/credit)"
    else:
        val, side = parse_amounts(d[m["amount"]])
        out["amount"] = val.abs()
        if m.get("type"):
            t = d[m["type"]].astype(str).str.strip().str.lower()
            type_hint = np.where(
                t.str.contains(r"\b(?:dr|debit|withdraw|expense|sent|paid)", regex=True),
                "Expense",
                np.where(
                    t.str.contains(r"\b(?:cr|credit|deposit|income|received)", regex=True),
                    "Income",
                    "",
                ),
            )
            typ = np.where(
                type_hint != "",
                type_hint,
                np.where(side != "", side, np.where(val < 0, "Expense", np.where(val > 0, positive_amount_type, ""))),
            )
        else:
            typ = np.where(
                side != "",
                side,
                np.where(val < 0, "Expense", np.where(val > 0, positive_amount_type, "")),
            )
        out["transaction_type"] = typ
        reason[(reason == "") & ~(out.amount > 0)] = "invalid amount"
        reason[(reason == "") & (out.transaction_type == "")] = "unknown transaction type"
    out["ref"] = d[m["ref"]].fillna("").astype(str).str.strip() if m.get("ref") else ""

    bad = d[reason != ""].copy()
    bad["reason"] = reason[reason != ""]
    ok = out[reason == ""].copy()
    ok["amount"] = ok.amount.round(2)

    dup_in_file = 0
    if m.get("ref"):  # a bank reference makes in-file duplicates certain
        has = ok.ref.ne("")
        dups = ok[has].duplicated(["ref", "transaction_date", "amount", "transaction_type"], keep="first")
        dup_in_file = int(dups.sum())
        ok = ok.drop(dups[dups].index)
    ok["payment_method"] = ok.description.map(payment_method)
    ok = ok.sort_values("transaction_date", kind="stable").reset_index(drop=True)
    ok = add_hash(ok)

    missing = int(out.transaction_date.isna().sum() + (out.amount.isna()).sum() + (out.description == "").sum())
    stats = dict(total_rows=n_rows, valid_rows=len(ok), invalid_rows=len(bad), duplicates_in_file=dup_in_file,
                 missing_values=missing, invalid_reasons=bad["reason"].value_counts().to_dict() if len(bad) else {})
    return ok, bad, stats


def add_hash(df):
    """Hash = date|description|amount|type|ref|nth-occurrence. The occurrence counter keeps two genuine identical
    payments on one day, while re-uploading an overlapping statement still deduplicates against the database."""
    key = (df.transaction_date.dt.strftime("%Y-%m-%d %H:%M") + "|" + df.description.str.upper() + "|"
           + df.amount.map(lambda x: f"{x:.2f}") + "|" + df.transaction_type + "|" + df.ref)
    occ = key.groupby(key).cumcount()
    df = df.copy()
    df["txn_hash"] = [hashlib.sha1(f"{k}|{o}".encode()).hexdigest() for k, o in zip(key, occ)]
    return df
