"""
app.py  -  ComplaintLens AI (Streamlit web interface)

Run with:   streamlit run app.py

Pages: Home | Analyze Complaint | Dashboard | Complaint History | About
The heavy lifting is done in complaint_analyzer.py (AI) and database.py (storage).
"""

import html
import logging
import os
from pathlib import Path

import streamlit as st

# set_page_config must be the first Streamlit command in the script.
st.set_page_config(
    page_title="ComplaintLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Import the rest. If a package is missing, explain it nicely. ----------
try:
    import pandas as pd
    import plotly.express as px
    from dotenv import load_dotenv

    load_dotenv()  # reads settings from the .env file (if it exists)

    from complaint_analyzer import (
        CATEGORIES,
        DATA_PATH,
        PRIORITY_LEVELS,
        SENTIMENTS,
        ComplaintAnalyzer,
        InvalidComplaintError,
    )
    from database import (
        DISPLAY_COLUMNS,
        DatabaseError,
        clear_user_complaints,
        init_db,
        load_complaints,
        save_complaint,
        seed_sample_data,
    )
except ImportError as import_error:
    st.error(
        "A required Python package is missing. Activate your virtual environment "
        "and run:  `pip install -r requirements.txt`"
    )
    st.caption(f"Details: {import_error}")
    st.stop()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING").upper())
logger = logging.getLogger("complaintlens")

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
PRIORITY_COLORS = {"Low": "#16a34a", "Medium": "#ca8a04", "High": "#ea580c", "Critical": "#dc2626"}
SENTIMENT_COLORS = {"Positive": "#16a34a", "Neutral": "#64748b", "Negative": "#dc2626"}
BRAND_COLOR = "#4338ca"

CUSTOM_CSS = """
<style>
.hero {background: linear-gradient(135deg, #4338ca 0%, #0d9488 100%); padding: 1.8rem 2.2rem;
       border-radius: 18px; color: #ffffff; margin-bottom: 1.2rem;}
.hero h1 {color: #ffffff; margin: 0; font-size: 2.1rem; line-height: 1.2;}
.hero p {margin: .4rem 0 0 0; font-size: 1.05rem; opacity: .95;}
.lens-card {border: 1px solid rgba(128,128,128,.30); border-radius: 14px; padding: .9rem 1.1rem;
            background: rgba(128,128,128,.07); height: 100%;}
.lens-label {font-size: .75rem; letter-spacing: .07em; text-transform: uppercase; opacity: .7;}
.lens-value {font-size: 1.35rem; font-weight: 700; margin-top: .15rem;}
.pill {display: inline-block; padding: .22rem .8rem; margin: .15rem .3rem .15rem 0; border-radius: 999px;
       background: rgba(67,56,202,.14); border: 1px solid rgba(67,56,202,.45); font-size: .92rem;}
.step-num {display: inline-block; width: 1.9rem; height: 1.9rem; line-height: 1.9rem; text-align: center;
           border-radius: 50%; background: #4338ca; color: #ffffff; font-weight: 700; margin-bottom: .4rem;}
.section-title {font-size: 1.05rem; font-weight: 700; margin: 1.1rem 0 .3rem 0;}
</style>
"""


def show_stretched(func, *args, **kwargs):
    """Call a Streamlit element so that it fills the page width.
    Newer Streamlit uses width="stretch", older versions use use_container_width=True."""
    last_error = None
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return func(*args, **kwargs, **extra)
        except Exception as exc:  # try the next option
            last_error = exc
    raise last_error


def escape_md(text: str) -> str:
    """Stop '$' from being treated as a maths symbol by Streamlit."""
    return str(text).replace("$", "\\$")


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def info_card(label: str, value: str, color: str = BRAND_COLOR) -> str:
    return (
        f'<div class="lens-card"><div class="lens-label">{html.escape(label)}</div>'
        f'<div class="lens-value" style="color:{color}">{html.escape(str(value))}</div></div>'
    )


# ---------------------------------------------------------------------------
# Start-up: load the AI engine and prepare the database
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the AI models (first run can take a few seconds)...")
def load_analyzer() -> ComplaintAnalyzer:
    """Created once and reused, so the model is not reloaded on every click."""
    return ComplaintAnalyzer()


def prepare_database(analyzer) -> None:
    """Create the table and load the sample data the first time the app runs."""
    if st.session_state.get("db_ready"):
        return
    try:
        init_db()
        added = seed_sample_data(analyzer)
        st.session_state["db_ready"] = True
        if added:
            logger.info("Loaded %s sample complaints.", added)
    except DatabaseError as exc:
        st.session_state["db_ready"] = False
        st.sidebar.warning(str(exc))


# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------
def page_home() -> None:
    hero("ComplaintLens AI", "AI-powered complaint analysis and intelligent resolution system.")

    st.write(
        "ComplaintLens AI reads a customer complaint, understands it using Natural Language "
        "Processing (NLP) and Machine Learning, and tells the support team **what the problem is, "
        "how serious it is and what to do next**. It replaces slow manual sorting with an instant, "
        "consistent first review."
    )

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Submit", "Type or paste a customer complaint on the Analyze page."),
        ("2", "Analyze", "NLP + ML find the category, sentiment, priority, urgency and keywords."),
        ("3", "Act", "Get a short summary and a recommended action. Everything is saved to the dashboard."),
    ]
    for column, (number, title, text) in zip(st.columns(3), steps):
        with column:
            st.markdown(
                f'<div class="lens-card"><span class="step-num">{number}</span>'
                f'<div class="lens-value" style="font-size:1.1rem">{title}</div>'
                f"<div>{text}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">What you get</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(
            "- 🗂️ **Category detection** (8 categories)\n"
            "- 😠 **Sentiment analysis** (positive / neutral / negative)\n"
            "- 🚦 **Priority and urgency** levels (Low to Critical)\n"
            "- 🔑 **Keyword extraction**"
        )
    with right:
        st.markdown(
            "- 📝 **Short complaint summary**\n"
            "- ✅ **Recommended action** for the support team\n"
            "- 📊 **Analytics dashboard** with interactive charts\n"
            "- 📋 **Searchable complaint history**"
        )

    st.write("")
    st.button(
        "📝 Analyze a complaint now",
        type="primary",
        on_click=lambda: st.session_state.update(page=PAGES[1]),
    )


# ---------------------------------------------------------------------------
# PAGE: Analyze Complaint
# ---------------------------------------------------------------------------
EXAMPLE_PLACEHOLDER = "-- choose an example --"
EXAMPLES = [
    "My money was deducted from my account but the payment failed.",
    "My order has not arrived even though it was supposed to be delivered yesterday.",
    "My account has been hacked and money was stolen.",
    "The product I received is damaged and the screen is cracked.",
    "The app keeps crashing every time I try to open it.",
    "I called customer care five times and nobody has solved my problem.",
    "I returned the shoes a month ago but I still have not received my refund.",
]


def use_example() -> None:
    """Copy the chosen example into the text box."""
    choice = st.session_state.get("example_choice")
    if choice and choice != EXAMPLE_PLACEHOLDER:
        st.session_state["complaint_text"] = choice


def render_result(result: dict) -> None:
    """Show the analysis result as cards, keyword pills and text boxes."""
    st.markdown("---")
    st.subheader("Analysis Result")

    cards = [
        ("Category", result["category"], BRAND_COLOR),
        ("Sentiment", result["sentiment"], SENTIMENT_COLORS.get(result["sentiment"], BRAND_COLOR)),
        ("Priority", result["priority"], PRIORITY_COLORS.get(result["priority"], BRAND_COLOR)),
        ("Urgency", result["urgency"], PRIORITY_COLORS.get(result["urgency"], BRAND_COLOR)),
    ]
    for column, (label, value, color) in zip(st.columns(4), cards):
        column.markdown(info_card(label, value, color), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Keywords</div>', unsafe_allow_html=True)
    if result["keywords"]:
        pills = "".join(f'<span class="pill">{html.escape(k)}</span>' for k in result["keywords"])
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.caption("No keywords found.")

    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
    st.info(escape_md(result["summary"]))

    st.markdown('<div class="section-title">Recommended Action</div>', unsafe_allow_html=True)
    st.success(escape_md(result["recommended_action"]))

    with st.expander("🔍 How was this decided?"):
        st.write(
            f"**Category confidence:** {result['category_confidence']:.0%}  \n"
            f"**Sentiment score:** {result['sentiment_score']} (-1 = very negative, +1 = very positive)  \n"
            f"**AI engine:** {result['model_status']}"
        )
        st.write(f"**Priority score: {result['priority_score']}**  (Low < 2, Medium 2-3, High 4-6, Critical 7+)")
        for reason in result["priority_reasons"] or ["No risk signals found."]:
            st.write(f"- {escape_md(reason)}")
        st.write(f"**Urgency score: {result['urgency_score']}**  (Low < 2, Medium 2-3, High 4-6, Critical 7+)")
        for reason in result["urgency_reasons"] or ["No urgency signals found."]:
            st.write(f"- {escape_md(reason)}")

    if result.get("complaint_id"):
        st.caption(f"Saved to Complaint History as #{result['complaint_id']}.")


def page_analyze(analyzer: ComplaintAnalyzer) -> None:
    hero("COMPLAINTLENS AI", "Intelligent Complaint Analysis")

    st.selectbox(
        "Need an idea? Try an example",
        [EXAMPLE_PLACEHOLDER] + EXAMPLES,
        key="example_choice",
        on_change=use_example,
    )
    text = st.text_area(
        "Enter your complaint:",
        key="complaint_text",
        height=150,
        max_chars=2000,
        placeholder="Example: My order has not arrived even though it was supposed to be delivered yesterday.",
    )

    if st.button("Analyze Complaint", type="primary"):
        try:
            with st.spinner("Analyzing your complaint..."):
                result = analyzer.analyze(text)
        except InvalidComplaintError as exc:
            st.session_state.pop("last_result", None)
            st.warning(str(exc))
        except Exception:  # never show technical details to normal users
            logger.exception("Analysis failed")
            st.session_state.pop("last_result", None)
            st.error("Sorry, something went wrong while analyzing the complaint. Please try again.")
        else:
            try:
                result["complaint_id"] = save_complaint(result)
            except DatabaseError as exc:
                st.warning(str(exc))
            st.session_state["last_result"] = result

    if st.session_state.get("last_result"):
        render_result(st.session_state["last_result"])


# ---------------------------------------------------------------------------
# PAGE: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    hero("Dashboard", "Live statistics about all analyzed complaints.")

    include_sample = st.checkbox("Include the sample dataset", value=True)
    try:
        df = load_complaints(include_sample)
    except DatabaseError as exc:
        st.error(str(exc))
        return

    df = df.dropna(subset=["created_at"])
    if df.empty:
        st.info("No complaints yet. Go to **Analyze Complaint** and submit one, or tick the sample dataset box above.")
        return

    # ---- KPI cards ----
    top_category = df["category"].mode().iloc[0]
    kpis = st.columns(5)
    kpis[0].metric("Total complaints", len(df))
    kpis[1].metric("High priority", int((df["priority"] == "High").sum()))
    kpis[2].metric("Critical", int((df["priority"] == "Critical").sum()))
    kpis[3].metric("Negative", int((df["sentiment"] == "Negative").sum()))
    kpis[4].metric("Top category", top_category)

    st.write("")

    # ---- Charts ----
    by_category = df["category"].value_counts().rename_axis("category").reset_index(name="count")
    fig_category = px.bar(
        by_category, x="count", y="category", orientation="h", text="count",
        title="Complaints by category", color_discrete_sequence=[BRAND_COLOR],
    )
    fig_category.update_layout(yaxis={"categoryorder": "total ascending", "title": ""}, xaxis_title="Complaints")

    by_priority = (
        df["priority"].value_counts().reindex(PRIORITY_LEVELS, fill_value=0)
        .rename_axis("priority").reset_index(name="count")
    )
    fig_priority = px.bar(
        by_priority, x="priority", y="count", color="priority", text="count",
        color_discrete_map=PRIORITY_COLORS, category_orders={"priority": PRIORITY_LEVELS},
        title="Complaints by priority",
    )
    fig_priority.update_layout(showlegend=False, xaxis_title="", yaxis_title="Complaints")

    by_sentiment = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    fig_sentiment = px.pie(
        by_sentiment, names="sentiment", values="count", hole=0.45,
        color="sentiment", color_discrete_map=SENTIMENT_COLORS, title="Sentiment distribution",
    )

    daily = df.set_index("created_at").resample("D").size().rename("complaints").reset_index()
    fig_trend = px.line(
        daily, x="created_at", y="complaints", markers=True,
        title="Complaint trends (per day)", color_discrete_sequence=[BRAND_COLOR],
    )
    fig_trend.update_layout(xaxis_title="Date", yaxis_title="Complaints")

    row1 = st.columns(2)
    with row1[0]:
        show_stretched(st.plotly_chart, fig_category)
    with row1[1]:
        show_stretched(st.plotly_chart, fig_priority)
    row2 = st.columns(2)
    with row2[0]:
        show_stretched(st.plotly_chart, fig_sentiment)
    with row2[1]:
        show_stretched(st.plotly_chart, fig_trend)

    # ---- Latest serious complaints ----
    st.markdown('<div class="section-title">🚨 Latest Critical and High priority complaints</div>', unsafe_allow_html=True)
    serious = df[df["priority"].isin(["Critical", "High"])].head(8)
    table = serious[["complaint_text", "category", "priority", "urgency", "created_at"]].copy()
    table["created_at"] = table["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    table.columns = ["Complaint", "Category", "Priority", "Urgency", "Date/Time"]
    show_stretched(st.dataframe, table, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE: Complaint History
# ---------------------------------------------------------------------------
def page_history() -> None:
    hero("Complaint History", "Browse, filter and download previously analyzed complaints.")

    include_sample = st.checkbox("Include the sample dataset", value=True, key="history_sample")
    try:
        df = load_complaints(include_sample)
    except DatabaseError as exc:
        st.error(str(exc))
        return
    if df.empty:
        st.info("No complaints saved yet. Analyze a complaint first.")
        return

    # ---- Filters ----
    col1, col2, col3 = st.columns(3)
    categories = col1.multiselect("Category", CATEGORIES)
    priorities = col2.multiselect("Priority", PRIORITY_LEVELS)
    sentiments = col3.multiselect("Sentiment", SENTIMENTS)
    search = st.text_input("Search in complaint text", placeholder="e.g. refund")

    filtered = df
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if priorities:
        filtered = filtered[filtered["priority"].isin(priorities)]
    if sentiments:
        filtered = filtered[filtered["sentiment"].isin(sentiments)]
    if search.strip():
        filtered = filtered[filtered["complaint_text"].str.contains(search.strip(), case=False, regex=False, na=False)]

    st.caption(f"Showing {len(filtered)} of {len(df)} complaints")
    if filtered.empty:
        st.warning("No complaints match these filters.")
    else:
        display = filtered[list(DISPLAY_COLUMNS)].rename(columns=DISPLAY_COLUMNS)
        display["Date/Time"] = display["Date/Time"].dt.strftime("%Y-%m-%d %H:%M")
        show_stretched(st.dataframe, display, hide_index=True)
        st.download_button(
            "⬇️ Download as CSV",
            display.to_csv(index=False).encode("utf-8"),
            file_name="complaint_history.csv",
            mime="text/csv",
        )

    with st.expander("Delete my complaints"):
        st.write("This removes only the complaints you submitted. The sample dataset is kept.")
        confirmed = st.checkbox("Yes, I understand")
        if st.button("Delete my complaints", disabled=not confirmed):
            try:
                clear_user_complaints()
                st.session_state.pop("last_result", None)
                st.success("Your complaints were deleted.")
                st.rerun()
            except DatabaseError as exc:
                st.error(str(exc))


# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
def page_about(analyzer: ComplaintAnalyzer) -> None:
    hero("About ComplaintLens AI", "An academic BTech CSE project on NLP and Machine Learning.")

    st.write(
        "ComplaintLens AI automatically analyses customer complaints so that support teams can "
        "sort, prioritise and answer them faster. It works completely offline, so no paid API key is needed."
    )

    st.markdown('<div class="section-title">Technology stack</div>', unsafe_allow_html=True)
    stack = pd.DataFrame(
        [
            ("Streamlit", "Web interface and dashboard"),
            ("Pandas / NumPy", "Data handling"),
            ("NLTK", "Tokenization, stop-words, stemming"),
            ("Scikit-learn", "TF-IDF + Logistic Regression category model"),
            ("Transformers + PyTorch", "Optional DistilBERT sentiment model"),
            ("Plotly", "Interactive charts"),
            ("SQLite", "Complaint history storage"),
            ("python-dotenv", "Settings from the .env file"),
        ],
        columns=["Technology", "Used for"],
    )
    show_stretched(st.dataframe, stack, hide_index=True)

    st.markdown('<div class="section-title">AI engine status</div>', unsafe_allow_html=True)
    info = analyzer.get_model_info()
    accuracy = info.get("cv_accuracy")
    st.write(
        f"- **Category model:** {info.get('algorithm', 'Keyword rules')} - {info['status']}\n"
        f"- **Training samples:** {info.get('training_samples', 'n/a')}\n"
        f"- **Cross-validation accuracy (ML model alone):** "
        f"{f'{accuracy:.0%}' if accuracy is not None else 'n/a'}\n"
        f"- **Sentiment engine:** "
        f"{'Transformers (DistilBERT)' if info['transformers'] else 'Lexicon-based (offline)'}\n"
        f"- **Last trained:** {info.get('trained_at', 'n/a')}"
    )
    st.caption(
        "The sample dataset is small (made for demonstration), so accuracy on very unusual complaints "
        "will be lower than on common ones. Add more labelled rows to data/complaints.csv, delete "
        "models/category_model.joblib and restart the app to retrain."
    )

    st.markdown('<div class="section-title">Future enhancements</div>', unsafe_allow_html=True)
    st.markdown(
        "- Multilingual complaint analysis\n- Email notifications\n- Admin login\n"
        "- Real-time analytics\n- Advanced ML models\n- Voice complaint submission"
    )


# ---------------------------------------------------------------------------
# MAIN: sidebar navigation + page routing
# ---------------------------------------------------------------------------
PAGES = ["🏠 Home", "📝 Analyze Complaint", "📊 Dashboard", "📋 Complaint History", "ℹ️ About"]


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    try:
        analyzer = load_analyzer()
    except Exception:
        logger.exception("Could not start the AI engine")
        st.error("The AI engine could not be started. Please check the installation and try again.")
        st.stop()

    prepare_database(analyzer)

    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=90)
        st.markdown("## ComplaintLens AI")
        page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.markdown("---")
        st.caption(f"AI engine: {analyzer.model_status}")
        if not DATA_PATH.exists():
            st.warning("Sample dataset (data/complaints.csv) not found.")

    if page == PAGES[0]:
        page_home()
    elif page == PAGES[1]:
        page_analyze(analyzer)
    elif page == PAGES[2]:
        page_dashboard()
    elif page == PAGES[3]:
        page_history()
    else:
        page_about(analyzer)


main()
