# ComplaintLens AI
**Intelligent Complaint Analysis and Management System**

> *AI-powered complaint analysis and intelligent resolution system.*

---

## Description

ComplaintLens AI is a web application that reads a customer complaint and, in about a second, tells the support team:

| Output | Example for *"My money was deducted from my account but the payment failed."* |
|---|---|
| Category | Payment Issue |
| Sentiment | Negative |
| Priority | High |
| Urgency | High |
| Keywords | money, deducted, account, payment, failed |
| Summary | Their money was deducted from their account but the payment failed. |
| Recommended action | Verify the transaction and initiate refund investigation. |

It uses **Natural Language Processing (NLP)** to clean and understand the text and **Machine Learning** to classify it. Every analysed complaint is saved to a database and shown on an analytics dashboard.

The project runs **completely offline** on a normal laptop. No paid API key is required.

## Features

- Complaint submission page with an **Analyze Complaint** button and one-click example complaints
- **Category detection** - Payment, Delivery, Product, Account, Technical, Service, Refund, Other
- **Sentiment analysis** - Positive / Neutral / Negative
- **Priority detection** - Low / Medium / High / Critical (financial loss, safety words, security threats, repeated problems, service disruption...)
- **Urgency detection** - Low / Medium / High / Critical
- **Keyword extraction** - words and phrases such as "credit card" and "bank account"
- **Complaint summary** - short third-person summary
- **Recommended action** based on category and priority (32 different actions)
- **"How was this decided?"** panel that shows the points behind every priority and urgency level
- **Dashboard** - KPIs and four Plotly charts (by category, by priority, sentiment, daily trend)
- **Complaint History** - filter by category, priority, sentiment, search by text, download as CSV
- **Sample dataset** of 84 realistic complaints so the dashboard is never empty
- Friendly error messages and automatic fallbacks (see *Error handling*)

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| Streamlit | Web interface and dashboard |
| Pandas, NumPy | Data processing |
| NLTK | Tokenization, stop-word removal, stemming |
| Scikit-learn | TF-IDF + Logistic Regression category classifier |
| Transformers + PyTorch | *Optional* DistilBERT sentiment model |
| Plotly | Interactive charts |
| python-dotenv | Settings from the `.env` file |
| SQLite (built into Python) | Complaint history storage |

## Project Structure

```text
ComplaintLens-AI/
|-- app.py                  Streamlit interface (all 5 pages)
|-- complaint_analyzer.py   NLP + ML engine
|-- database.py             SQLite storage
|-- requirements.txt        Dependencies
|-- README.md
|-- .env.example            Settings template
|-- data/
|   `-- complaints.csv      84 labelled sample complaints (also the training data)
|-- models/                 Trained model is saved here automatically
|-- assets/
|   `-- logo.png
`-- tests/
    `-- test_complaint_analyzer.py   Automated tests
```

## Installation

You need **Python 3.12** and **VS Code**. Open the project folder in VS Code, open the terminal (**Terminal > New Terminal**, PowerShell) and run:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell says *"running scripts is disabled on this system"*, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

> `torch` and `transformers` are large (several hundred MB). They are **optional** at run time. If the download is too slow, delete those two lines from `requirements.txt` before installing - the app works the same way.

*(Optional)* For NLTK's full stop-word list: `python -c "import nltk; nltk.download('stopwords')"`. The app has a built-in list, so this is not required.

## Running the Project

```powershell
streamlit run app.py
```

Your browser opens `http://localhost:8501`. The first start takes a few seconds because the app trains the category model (`models/category_model.joblib`) and loads the sample data into `data/complaints.db`. Later starts are instant.

## Expected Output

1. **Sidebar** with the logo and five pages: Home, Analyze Complaint, Dashboard, Complaint History, About.
2. **Home** - a colourful banner, a short explanation and a *How it works* section.
3. **Analyze Complaint** - type a complaint (or pick an example), click **Analyze Complaint**, and see four coloured cards (Category, Sentiment, Priority, Urgency), keyword pills, a summary and a recommended action.
4. **Dashboard** - five KPI numbers (total, high priority, critical, negative, top category) and four charts.
5. **Complaint History** - a table of every complaint with filters and a CSV download button.
6. **About** - technology list and the status and accuracy of the AI model.

## Testing

```powershell
python -m unittest discover -s tests -v
```

Try these in the app:

| Complaint | Expected result |
|---|---|
| My account has been hacked and money was stolen. | Account Issue, **Critical** |
| My money was deducted from my account but the payment failed. | Payment Issue, High priority, High urgency |
| My order has not arrived even though it was supposed to be delivered yesterday. | Delivery Issue, Medium |
| The mixer grinder makes a burning smell and sparks when switched on | Product Issue, **Critical** (safety) |
| Thank you for the quick help! | Positive, category Other |
| *(empty box)* | Friendly warning, nothing crashes |

## How It Works (short version)

1. **Preprocessing** - lowercase, remove symbols, tokenize (NLTK), remove stop-words, stem.
2. **Category** - TF-IDF + Logistic Regression (60 % weight) combined with keyword rules (40 % weight).
3. **Sentiment** - word lists with negation handling ("not helpful" is negative) and extra weight after "but".
4. **Priority / Urgency** - clues such as financial loss, safety words or security threats add points; the total becomes Low / Medium / High / Critical. The points are shown to the user, so every decision is explainable.
5. **Keywords** - known phrases first, then the highest-scoring domain words.
6. **Summary** - the most informative sentence(s), rewritten in third person.
7. **Action** - looked up from the (category, priority) table.

## Settings (`.env`)

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

| Setting | Meaning |
|---|---|
| `USE_TRANSFORMERS=false` | Offline lexicon sentiment (default, fast) |
| `USE_TRANSFORMERS=true` | DistilBERT sentiment. Needs internet once to download the model (~270 MB). Falls back automatically if it fails. |
| `LOG_LEVEL=WARNING` | Amount of technical detail printed in the terminal |

## Error Handling

| Situation | What happens |
|---|---|
| Empty, too short, too long or symbol-only complaint | Yellow warning that explains what to fix |
| `data/complaints.csv` missing | Sidebar warning; analysis still works using keyword rules |
| Model file missing or corrupted | Model is retrained automatically; if that is impossible, keyword rules are used |
| Package missing | App shows `pip install -r requirements.txt` instructions |
| Database error | Friendly message; the analysis result is still shown |
| Transformers not installed / no internet | Silent fallback to the offline sentiment engine |

Technical details are written to the terminal log only, never shown to normal users.

## Common Problems

| Problem | Fix |
|---|---|
| `'streamlit' is not recognized` | Activate the environment first: `venv\Scripts\activate` |
| `ModuleNotFoundError` | `pip install -r requirements.txt` inside the activated environment |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Want to retrain after editing the CSV | Delete `models\category_model.joblib` and restart the app |
| Dashboard shows old numbers | Delete `data\complaints.db` and restart (sample data is reloaded) |
| Wrong Python version | `python --version` must show 3.12.x |

## Limitations

- The training set is small (84 rows), so the category model alone scores about 60 % in cross-validation. The keyword rules lift accuracy on typical complaints, but unusual wording can still be mislabelled. Adding more labelled rows to `data/complaints.csv` is the best way to improve it.
- Priority and urgency use hand-written, explainable rules (not a trained model). The weights were tuned on the sample complaints, so treat the results as a decision aid, not a final decision.
- Sentiment is lexicon-based by default; sarcasm is not detected.
- English only.

## Future Enhancements

- Multilingual complaint analysis
- Email notifications
- Admin login
- Real-time analytics
- Advanced ML models
- Voice complaint submission
