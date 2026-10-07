"""Streamlit UI for the expense tracker."""
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__:
    from .config import ALL_CATS, EXCLUDED, SAMPLE_PATH
    from .ingest import clean, detect_columns, parse_amounts, read_file
    from .analytics import category_table, kpis, merchant_table, month_label, monthly, prep
    from .categorize import MLClassifier, categorize_df
    from .db import (
        add_upload,
        init as init_db,
        insert_transactions,
        load as load_transactions,
        merchant_map,
        update_transaction,
    )
else:
    from config import ALL_CATS, EXCLUDED, SAMPLE_PATH
    from ingest import clean, detect_columns, parse_amounts, read_file
    from analytics import category_table, kpis, merchant_table, month_label, monthly, prep
    from categorize import MLClassifier, categorize_df
    from db import (
        add_upload,
        init as init_db,
        insert_transactions,
        load as load_transactions,
        merchant_map,
        update_transaction,
    )


st.set_page_config(page_title="Expense Tracker", page_icon="💸", layout="wide")


def _sample_file() -> Path:
    base = Path(__file__).resolve().parent
    candidates = [
        SAMPLE_PATH,
        base / "sample_upi_statement.csv",
        base / "data" / "sample" / "sample_upi_statement.csv",
    ]
    for path in candidates:
        if path and path.exists():
            return path
    return base / "sample_upi_statement.csv"


def _normalise_cell(value):
    return "" if pd.isna(value) else str(value).strip()


def _apply_positive_amount_direction(raw, mapping, positive_amount_type):
    amount_column = mapping.get("amount")
    if not amount_column or mapping.get("debit") or mapping.get("credit"):
        return raw

    values, side_hints = parse_amounts(raw[amount_column])
    direction = pd.Series("", index=raw.index, dtype="object")
    if mapping.get("type"):
        type_values = raw[mapping["type"]].fillna("").astype(str).str.strip().str.lower()
        expense_marker = type_values.str.contains(
            r"\b(?:dr|debit|withdraw|expense|sent|paid)", regex=True
        )
        income_marker = type_values.str.contains(
            r"\b(?:cr|credit|deposit|income|received)", regex=True
        )
        has_direction = expense_marker | income_marker
        fallback_rows = ~has_direction
        direction.loc[expense_marker] = "Expense"
        direction.loc[income_marker] = "Income"
        direction.loc[fallback_rows & side_hints.eq("Expense")] = "Expense"
        direction.loc[fallback_rows & side_hints.eq("Income")] = "Income"
        direction.loc[fallback_rows & values.lt(0)] = "Expense"
        direction.loc[fallback_rows & values.gt(0)] = positive_amount_type
        needs_direction = fallback_rows & direction.ne("")
        if not needs_direction.any():
            return raw

        prepared = raw.copy()
        prepared.loc[needs_direction, mapping["type"]] = direction.loc[needs_direction].map(
            {"Expense": "Debit", "Income": "Credit"}
        )
        return prepared

    if positive_amount_type != "Expense":
        return raw

    expense_rows = values.gt(0) & side_hints.eq("")
    if not expense_rows.any():
        return raw

    prepared = raw.copy()
    prepared.loc[expense_rows, amount_column] = (
        "-" + raw.loc[expense_rows, amount_column].astype(str).str.strip()
    )
    return prepared


init_db()

st.title("Expense tracker")
st.caption("Upload a bank or UPI statement, review detected columns, and explore your spending.")

if "page_notice" in st.session_state:
    st.success(st.session_state.pop("page_notice"))

uploaded = st.sidebar.file_uploader(
    "Upload a bank or UPI statement",
    type=["csv", "xlsx", "xls", "pdf"],
    help="Supported formats: CSV, Excel, or PDF. PDF statements are read locally with OCR.",
)
saved_transactions = load_transactions()
bad_rows = pd.DataFrame()
prepared = pd.DataFrame()
import_notice = ""
did_import = False

if uploaded is not None:
    upload_bytes = uploaded.getvalue()
    upload_digest = hashlib.sha256(upload_bytes).hexdigest()
    try:
        if uploaded.name.lower().endswith(".pdf"):
            progress = st.progress(0, text="Reading PDF locally with OCR...")
            try:
                raw = read_file(
                    uploaded.name,
                    upload_bytes,
                    progress_callback=lambda page, total: progress.progress(
                        page / total,
                        text=f"Reading PDF locally with OCR (page {page} of {total})...",
                    ),
                )
            finally:
                progress.empty()
        else:
            raw = read_file(uploaded.name, upload_bytes)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    detected = detect_columns(raw)
    column_names = list(raw.columns)
    no_column = "— Not in this file —"
    column_fields = [
        ("date", "Transaction date *", "The date the payment was made."),
        ("description", "Description or payee *", "The transaction details, narration, or merchant name."),
        ("amount", "Amount", "Use this when your file has one amount column. If it has separate debit and credit columns, leave this unset."),
        ("type", "Transaction type", "Use this when amounts are unsigned and a column says Debit/Credit or Expense/Income."),
        ("debit", "Money out / debit", "Use this with Money in / credit when your file has separate withdrawal and deposit columns."),
        ("credit", "Money in / credit", "Use this with Money out / debit when your file has separate withdrawal and deposit columns."),
        ("ref", "Reference number", "Optional: helps identify repeated transactions."),
        ("balance", "Account balance", "Optional: not used in spending calculations."),
    ]
    st.subheader("Step 1: Match your statement columns")
    st.caption("Choose the file column that matches each item. Fields marked * are required.")
    with st.form("column_mapping_form"):
        selected = {}
        for key, label, help_text in column_fields:
            options = [no_column, *column_names]
            current = detected.get(key) or no_column
            index = options.index(current) if current in options else 0
            selected[key] = st.selectbox(
                label,
                options,
                index=index,
                help=help_text,
                key=f"upload_column_{key}_{upload_digest[:10]}",
            )
        positive_amount_handling = st.selectbox(
            "If an amount is positive but has no debit/credit marker",
            [
                "Treat as money out / expense",
                "Use statement direction (positive = money in)",
                "Treat as money in / income",
            ],
            index=0,
            help=(
                "Bank and UPI spending files often show all positive values as outgoing payments. "
                "Signed amounts and recognized debit/credit values still take priority."
            ),
            key=f"upload_positive_amount_handling_{upload_digest[:10]}",
        )
        preview_submitted = st.form_submit_button(
            "Preview transactions",
            type="primary",
            icon=":material/preview:",
        )

    if preview_submitted:
        st.session_state.pop("upload_preview", None)
        mapping = {key: value for key, value in selected.items() if value != no_column}
        try:
            if positive_amount_handling == "Treat as money in / income":
                positive_amount_type = "Income"
            elif positive_amount_handling == "Use statement direction (positive = money in)":
                positive_amount_type = "Income"
            else:
                positive_amount_type = "Expense"
            clean_input = _apply_positive_amount_direction(
                raw,
                mapping,
                positive_amount_type,
            )
            cleaned, rejected, stats = clean(clean_input, mapping)
        except ValueError as exc:
            st.error(str(exc))
            cleaned = pd.DataFrame()
            rejected = pd.DataFrame()
            stats = {}

        if stats and cleaned.empty:
            st.warning("No transactions could be prepared. Check your column choices and try again.")
            if not rejected.empty:
                st.dataframe(
                    rejected.head(50),
                    hide_index=True,
                    alt="Rows rejected while preparing the statement import",
                )
        elif not cleaned.empty:
            classifier = MLClassifier.load_or_seed()
            classified = categorize_df(cleaned, user_map=merchant_map(), ml=classifier)
            st.session_state["upload_preview"] = {
                "digest": upload_digest,
                "filename": uploaded.name,
                "transactions": classified,
                "rejected": rejected,
                "stats": stats,
            }
            st.rerun()

    preview = st.session_state.get("upload_preview")
    if preview and preview["digest"] == upload_digest and not did_import:
        preview_data = preview["transactions"]
        type_counts = preview_data.transaction_type.value_counts()
        expense_count = int(type_counts.get("Expense", 0))
        income_count = int(type_counts.get("Income", 0))
        st.subheader("Step 2: Check your transactions")
        with st.container(horizontal=True):
            st.metric("Ready to import", f"{len(preview_data):,}")
            st.metric("Money out", f"{expense_count:,}")
            st.metric("Money in", f"{income_count:,}")
            st.metric("Rows skipped", f"{len(preview['rejected']):,}")

        if expense_count == 0:
            st.warning(
                "No outgoing payments were detected, so expense charts will be empty. "
                "Check the Transaction type column (for example, it should contain Debit/Credit), "
                "or map separate Money out and Money in columns, then preview again."
            )

        st.dataframe(
            preview_data[
                [
                    "transaction_date",
                    "description",
                    "merchant",
                    "category",
                    "amount",
                    "transaction_type",
                ]
            ].head(20),
            column_config={
                "transaction_date": st.column_config.DatetimeColumn("Date", format="MMM D, YYYY"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "merchant": st.column_config.TextColumn("Merchant"),
                "category": st.column_config.TextColumn("Suggested category"),
                "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                "transaction_type": st.column_config.TextColumn("Money direction"),
            },
            hide_index=True,
            alt="Preview of classified transactions before saving",
        )
        st.caption("If the money direction looks wrong, change the column choices above and preview again.")
        if st.button(
            "Import these transactions",
            type="primary",
            icon=":material/upload:",
        ):
            batch_id = add_upload(uploaded.name, preview["stats"]["valid_rows"])
            added = insert_transactions(preview_data, batch_id)
            import_notice = (
                f"Imported {added:,} new transactions from {uploaded.name}; "
                f"{preview['stats']['valid_rows'] - added:,} duplicates skipped and "
                f"{len(preview['rejected']):,} invalid rows rejected."
            )
            bad_rows = preview["rejected"]
            saved_transactions = load_transactions()
            st.session_state.pop("upload_preview", None)
            did_import = True
    elif (not preview or preview["digest"] != upload_digest) and not preview_submitted:
        st.info("Select the matching statement columns above, then preview before importing.")
        st.dataframe(
            raw.head(10),
            hide_index=True,
            alt="Preview of the uploaded statement columns",
        )
elif st.session_state.get("upload_preview"):
    st.session_state.pop("upload_preview", None)
elif saved_transactions.empty:
    sample_path = _sample_file()
    if not sample_path.exists():
        st.info("Upload a statement to get started. No sample data is available in this project.")
        st.stop()
    raw = pd.read_csv(sample_path, dtype=str)
    try:
        cleaned, bad_rows, _ = clean(raw, detect_columns(raw))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if cleaned.empty:
        st.warning("No valid transactions were found in the bundled sample statement.")
        st.stop()
    classifier = MLClassifier.load_or_seed()
    prepared = prep(categorize_df(cleaned, user_map=merchant_map(), ml=classifier))
    st.info("Showing the bundled sample statement. Upload your own statement to save and edit your transactions.")
else:
    prepared = prep(saved_transactions)

if uploaded is not None and not saved_transactions.empty:
    prepared = prep(saved_transactions)

if import_notice:
    st.success(import_notice)

if prepared.empty:
    st.info("No saved transactions yet. Upload a statement and confirm its columns to get started.")
    st.stop()

st.header("Monthly spending summary", icon=":material/analytics:")
available_months = sorted(prepared.month.dropna().unique(), reverse=True)
period_options = ["All months", *available_months]
selected_period = st.selectbox(
    "Choose a month",
    period_options,
    index=1 if available_months else 0,
    format_func=lambda value: month_label(value) if value != "All months" else value,
    help="Choose a month to update totals, categories, and merchants.",
)
summary_data = (
    prepared if selected_period == "All months"
    else prepared[prepared.month == selected_period]
)
summary_metrics = kpis(summary_data)

income_payments = summary_data[
    (summary_data.transaction_type == "Income")
    & ~summary_data.category.isin(EXCLUDED)
]
refund_payments = summary_data[
    (summary_data.transaction_type == "Refund")
    | (summary_data.category == "Refund")
]
with st.container(horizontal=True):
    with st.container(border=True):
        st.metric("Total spent", f"₹{summary_metrics['expense']:,.0f}")
        st.caption(f"{summary_metrics['n_expense']:,} payments")
    with st.container(border=True):
        st.metric("Total received", f"₹{summary_metrics['income']:,.0f}")
        st.caption(f"{len(income_payments):,} payments")
    with st.container(border=True):
        st.metric("Refunds", f"₹{refund_payments.amount.sum():,.0f}")
        st.caption(f"{len(refund_payments):,} payments")
    with st.container(border=True):
        st.metric("Transactions", f"{len(summary_data):,}")
        st.caption("In this period")
st.caption("Transfers and refunds are excluded from spending and income totals.")

category_data = category_table(summary_data)
merchant_data = merchant_table(summary_data)
income_only = summary_metrics["n_expense"] == 0 and summary_metrics["income"] > 0
if summary_metrics["n_expense"] == 0 and summary_metrics["n_txn"] > 0:
    st.warning(
        "No spending payments are recorded for this month. "
        + (
            "Incoming-payment charts are shown instead. "
            if income_only else ""
        )
        + "If a payment is marked incorrectly, change its type in the transaction table below."
    )

st.subheader("Categories and amounts", icon=":material/category:")
if income_only:
    category_chart_data = (
        summary_data[
            (summary_data.transaction_type == "Income")
            & ~summary_data.category.isin(EXCLUDED)
        ]
        .groupby("category", as_index=False)
        .amount.sum()
        .rename(columns={"amount": "total"})
        .sort_values("total", ascending=False)
    )
    category_heading = "Income by source"
    category_alt = "Incoming payment totals grouped by category"
    category_caption = "Incoming payments"
else:
    category_chart_data = category_data[["category", "total"]]
    category_heading = "Spending by category"
    category_alt = "Spending totals by category, highest amount first"
    category_caption = "Payments"

if category_chart_data.empty:
    st.info("No category totals are available for this period.")
else:
    st.caption(category_heading)
    st.bar_chart(
        category_chart_data.head(10),
        x="total",
        y="category",
        horizontal=True,
        x_label="Amount (₹)",
        y_label="Category",
        alt=category_alt,
    )
    if income_only:
        category_list = (
            summary_data[
                (summary_data.transaction_type == "Income")
                & ~summary_data.category.isin(EXCLUDED)
            ]
            .groupby("category")
            .amount.agg(total="sum", count="count")
            .reset_index()
        )
    else:
        category_list = category_data[["category", "count", "total"]].copy()
    category_list = category_list.rename(
        columns={"category": "Category", "count": category_caption, "total": "Amount"}
    )
    st.dataframe(
        category_list.head(10),
        column_config={
            "Category": st.column_config.TextColumn("Category"),
            category_caption: st.column_config.NumberColumn(category_caption, format="%d"),
            "Amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
        },
        hide_index=True,
        alt="Categories with payment counts and total amounts",
    )

spending_rows = summary_data[
    (summary_data.transaction_type == "Expense")
    & ~summary_data.category.isin(EXCLUDED)
]
st.subheader("Monthly category spending", icon=":material/summarize:")
st.caption(
    "See the number of payments and total amount for each category and subcategory. "
    "For example, Bills → Electricity shows all electricity bill payments in the selected period."
)
if spending_rows.empty:
    st.info("No expense payments are available for this period.")
else:
    monthly_category_totals = (
        spending_rows.assign(
            subcategory=spending_rows.subcategory.fillna("").replace("", "Other")
        )
        .groupby(["month", "category", "subcategory"], as_index=False)
        .agg(
            transactions=("amount", "size"),
            total_spent=("amount", "sum"),
            average_payment=("amount", "mean"),
        )
        .sort_values(["month", "total_spent"], ascending=[False, False])
        .reset_index(drop=True)
    )
    monthly_category_totals["month"] = monthly_category_totals["month"].map(month_label)
    monthly_category_totals = monthly_category_totals.rename(
        columns={
            "month": "Month",
            "category": "Category",
            "subcategory": "Subcategory",
            "transactions": "Transactions",
            "total_spent": "Total spent (₹)",
            "average_payment": "Average payment (₹)",
        }
    )
    st.dataframe(
        monthly_category_totals,
        column_config={
            "Month": st.column_config.TextColumn("Month"),
            "Category": st.column_config.TextColumn("Category"),
            "Subcategory": st.column_config.TextColumn("Subcategory"),
            "Transactions": st.column_config.NumberColumn("Transactions", format="%d"),
            "Total spent (₹)": st.column_config.NumberColumn(
                "Total spent (₹)", format="₹%.2f"
            ),
            "Average payment (₹)": st.column_config.NumberColumn(
                "Average payment (₹)", format="₹%.2f"
            ),
        },
        hide_index=True,
        alt="Monthly expense totals, payment counts, and average amounts by category and subcategory",
    )

st.subheader("Monthly merchant spending", icon=":material/store:")
st.caption(
    "See how much you spent at each merchant, how many payments you made, "
    "and the average amount per payment."
)
if spending_rows.empty:
    st.info("No merchant spending is available for this period.")
else:
    monthly_merchant_totals = (
        spending_rows.assign(
            merchant=spending_rows.merchant.fillna("").replace("", "Unknown merchant")
        )
        .groupby(["month", "merchant"], as_index=False)
        .agg(
            transactions=("amount", "size"),
            total_spent=("amount", "sum"),
            average_payment=("amount", "mean"),
        )
        .sort_values(["month", "total_spent"], ascending=[False, False])
        .reset_index(drop=True)
    )
    monthly_merchant_totals["month"] = monthly_merchant_totals["month"].map(month_label)
    monthly_merchant_totals = monthly_merchant_totals.rename(
        columns={
            "month": "Month",
            "merchant": "Merchant",
            "transactions": "Transactions",
            "total_spent": "Total spent (₹)",
            "average_payment": "Average payment (₹)",
        }
    )
    st.dataframe(
        monthly_merchant_totals,
        column_config={
            "Month": st.column_config.TextColumn("Month"),
            "Merchant": st.column_config.TextColumn("Merchant"),
            "Transactions": st.column_config.NumberColumn("Transactions", format="%d"),
            "Total spent (₹)": st.column_config.NumberColumn(
                "Total spent (₹)", format="₹%.2f"
            ),
            "Average payment (₹)": st.column_config.NumberColumn(
                "Average payment (₹)", format="₹%.2f"
            ),
        },
        hide_index=True,
        alt="Monthly expense totals, payment counts, and average amounts by merchant",
    )

st.subheader("Top merchants", icon=":material/storefront:")
if income_only:
    merchant_chart_data = (
        summary_data[
            (summary_data.transaction_type == "Income")
            & ~summary_data.category.isin(EXCLUDED)
        ]
        .groupby("merchant", as_index=False)
        .amount.sum()
        .rename(columns={"amount": "total"})
        .sort_values("total", ascending=False)
    )
    merchant_alt = "Incoming amounts grouped by source"
else:
    merchant_chart_data = merchant_data[["merchant", "total"]]
    merchant_alt = "Top merchants ranked by total spending"

if merchant_chart_data.empty:
    st.info("No merchant totals are available for this period.")
else:
    top_merchants = merchant_chart_data.head(10)
    st.bar_chart(
        top_merchants,
        x="total",
        y="merchant",
        horizontal=True,
        x_label="Amount (₹)",
        y_label="Merchant",
        alt=merchant_alt,
    )
    if not income_only:
        st.dataframe(
            merchant_data.head(10),
            column_config={
                "merchant": st.column_config.TextColumn("Merchant"),
                "total": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                "count": st.column_config.NumberColumn("Payments", format="%d"),
                "avg": st.column_config.NumberColumn("Average payment (₹)", format="₹%.2f"),
                "first": st.column_config.DateColumn("First payment"),
                "last": st.column_config.DateColumn("Last payment"),
                "recurring": st.column_config.CheckboxColumn("Recurring"),
            },
            hide_index=True,
            alt="Top merchants with amounts, payment counts, and payment dates",
        )

monthly_data = monthly(prepared)
if not monthly_data.empty:
    st.subheader("Monthly activity", icon=":material/calendar_month:")
    monthly_chart = monthly_data[["expense", "income"]].rename_axis("Month").reset_index()
    monthly_chart = monthly_chart.rename(
        columns={"expense": "Money out", "income": "Money in"}
    )
    countable = prepared[
        ~prepared.category.isin(EXCLUDED)
        & ~prepared.transaction_type.isin(EXCLUDED)
    ]
    monthly_counts = (
        countable.groupby(["month", "transaction_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Expense", "Income"], fill_value=0)
        .rename_axis("Month")
        .reset_index()
        .rename(columns={"Expense": "Outgoing payments", "Income": "Incoming payments"})
    )
    cashflow_col, volume_col = st.columns(2)
    with cashflow_col:
        with st.container(border=True):
            st.markdown("**Money in and out**")
            st.bar_chart(
                monthly_chart,
                x="Month",
                y=["Money out", "Money in"],
                x_label="Month",
                y_label="Amount (₹)",
                alt="Monthly outgoing and incoming payment totals",
            )
    with volume_col:
        with st.container(border=True):
            st.markdown("**Payment count**")
            st.bar_chart(
                monthly_counts,
                x="Month",
                y=["Outgoing payments", "Incoming payments"],
                x_label="Month",
                y_label="Number of payments",
                alt="Monthly counts of outgoing and incoming payments",
            )
    with st.expander("View monthly savings trend"):
        savings_chart = monthly_data[["savings"]].rename_axis("Month").reset_index()
        savings_chart = savings_chart.rename(columns={"savings": "Savings"})
        st.line_chart(
            savings_chart,
            x="Month",
            y="Savings",
            x_label="Month",
            y_label="Savings (₹)",
            alt="Monthly savings over time",
        )

st.subheader("Classified transactions")
st.caption(
    "Edit the amount, transaction type, merchant, category, subcategory, payment method, "
    "or review status. Changes are saved and recalculate the dashboard."
)

filter_col, search_col = st.columns([1, 2])
categories = sorted(prepared.category.dropna().astype(str).unique())
with filter_col:
    selected_category = st.selectbox("Filter by category", ["All categories", *categories])
with search_col:
    search_text = st.text_input(
        "Search transactions",
        type="search",
        placeholder="Search descriptions or merchants",
    )

filtered = prepared.sort_values("transaction_date", ascending=False).copy()
if selected_category != "All categories":
    filtered = filtered[filtered.category == selected_category]
if search_text.strip():
    query = search_text.strip()
    matches = (
        filtered.description.fillna("").astype(str).str.contains(query, case=False, regex=False)
        | filtered.merchant.fillna("").astype(str).str.contains(query, case=False, regex=False)
    )
    filtered = filtered[matches]

if filtered.empty:
    st.info("No transactions match these filters.")
else:
    editable_columns = [
        "amount",
        "merchant",
        "category",
        "subcategory",
        "transaction_type",
        "payment_method",
        "review_status",
    ]
    display_columns = [
        "transaction_date",
        "description",
        "merchant",
        "category",
        "subcategory",
        "amount",
        "transaction_type",
        "payment_method",
    ]
    editor_columns = {
        "transaction_date": st.column_config.DatetimeColumn("Date", format="MMM D, YYYY"),
        "description": st.column_config.TextColumn("Description", width="large"),
        "merchant": st.column_config.TextColumn("Merchant"),
        "category": st.column_config.SelectboxColumn(
            "Category",
            options=sorted(ALL_CATS),
            required=True,
            help="Choose the best matching transaction category.",
        ),
        "subcategory": st.column_config.SelectboxColumn(
            "Subcategory",
            options=sorted({sub for options in ALL_CATS.values() for sub in options}),
            required=True,
            help="Choose the detail that best describes this transaction.",
        ),
        "amount": st.column_config.NumberColumn(
            "Amount (₹)",
            min_value=0.01,
            step=0.01,
            format="₹%.2f",
            help="Enter a positive amount. Use transaction type to mark money in or out.",
        ),
        "transaction_type": st.column_config.SelectboxColumn(
            "Type",
            options=["Expense", "Income", "Refund", "Transfer"],
            required=True,
            help="Refunds and transfers are excluded from spending totals.",
        ),
        "payment_method": st.column_config.SelectboxColumn(
            "Payment method",
            options=["UPI", "IMPS", "NEFT", "RTGS", "ATM", "Card", "Auto-debit", "Other"],
            required=True,
        ),
        "review_status": st.column_config.SelectboxColumn(
            "Review status",
            options=["Unreviewed", "Normal", "Suspicious"],
            required=True,
            help="Mark a transaction for follow-up or confirm it as normal.",
        ),
    }

    if "id" in prepared.columns:
        editor_data = filtered.set_index("id")[display_columns + ["user_status"]]
        editor_data = editor_data.rename(columns={"user_status": "review_status"})
        editor_data["review_status"] = editor_data["review_status"].fillna("").astype(str).replace(
            {"": "Unreviewed", "normal": "Normal", "suspicious": "Suspicious"}
        )
        edited_data = st.data_editor(
            editor_data,
            key="classified_transactions_editor",
            hide_index=True,
            num_rows="fixed",
            disabled=[
                "transaction_date",
                "description",
            ],
            column_config=editor_columns,
            width="stretch",
            alt="Transaction table. Edit amounts, types, classifications, payment methods, and review status.",
        )
        changed_rows = []
        status_values = {
            "Unreviewed": "",
            "Normal": "normal",
            "Suspicious": "suspicious",
        }
        for txn_id, row in edited_data.iterrows():
            original = editor_data.loc[txn_id]
            changes = {}
            for column in editable_columns:
                original_value = original[column]
                if column == "review_status":
                    database_column = "user_status"
                    new_value = status_values[row[column]]
                    original_value = status_values[original_value]
                else:
                    database_column = column
                    new_value = row[column] if column == "amount" else _normalise_cell(row[column])
                    original_value = (
                        original_value
                        if column == "amount"
                        else _normalise_cell(original_value)
                    )
                if new_value != original_value:
                    changes[database_column] = new_value
            if changes:
                update_transaction(int(txn_id), **changes)
                changed_rows.append(txn_id)
        if changed_rows:
            st.session_state["page_notice"] = f"Saved edits to {len(changed_rows):,} transaction(s)."
            st.rerun()
        with st.expander("View classification details"):
            detail_columns = [
                column
                for column in [
                    "transaction_date",
                    "description",
                    "merchant",
                    "category",
                    "subcategory",
                    "category_confidence",
                    "category_source",
                    "category_reason",
                ]
                if column in filtered.columns
            ]
            if len(detail_columns) > 4:
                st.dataframe(
                    filtered[detail_columns],
                    column_config={
                        "transaction_date": st.column_config.DatetimeColumn(
                            "Date", format="MMM D, YYYY"
                        ),
                        "description": st.column_config.TextColumn(
                            "Description", width="large"
                        ),
                        "category_confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0,
                            max_value=1,
                            format="percent",
                        ),
                        "category_source": st.column_config.TextColumn("Method"),
                        "category_reason": st.column_config.TextColumn(
                            "Why this category", width="large"
                        ),
                    },
                    hide_index=True,
                    alt="Classification method and confidence for each transaction",
                )
    else:
        st.dataframe(
            filtered[display_columns],
            column_config=editor_columns,
            hide_index=True,
            width="stretch",
            alt="Sample classified transactions with dates, merchants, categories, and amounts",
        )

if not bad_rows.empty:
    with st.expander("Rejected rows"):
        st.dataframe(
            bad_rows.head(50),
            hide_index=True,
            alt="Rows rejected during the latest statement import",
        )
