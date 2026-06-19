# 🔬 DataLens AI — Instant EDA + LLM Insights

> Drop a CSV. Get interactive charts, correlation maps, data quality reports, and AI-powered insights — in under 10 seconds.

![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_LLaMA_3.3_70B-powered-8b5cf6?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)

---

## ✨ What it does

DataLens AI is a zero-config data analysis tool that transforms any CSV into a full analytical report:

| Feature | Details |
|---|---|
| **📋 Overview** | Dataset preview, numeric & categorical summaries, column types |
| **📊 Charts** | Histogram, Box, Violin, Line, Area, ECDF, Scatter with OLS trendline, Category bar |
| **🔥 Correlations** | Interactive heatmap (Pearson / Spearman / Kendall), top correlated pairs |
| **🧹 Data Quality** | Missing value map, duplicate detection, IQR outlier report |
| **🤖 AI Insights** | Groq LLaMA-3.3-70B analysis with custom questions or quick-fire templates |

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/<your-username>/datalens-ai.git
cd datalens-ai
pip install -r requirements.txt
```

### 2. Add your Groq API key

**Option A — Streamlit secrets (recommended for deployment)**
```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_..."
```

**Option B — Sidebar input** — paste your key directly in the app UI (never stored).

### 3. Run
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.22.0
groq>=0.9.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo + `app.py` as entry point
4. Add `GROQ_API_KEY` under **Secrets**
5. Deploy 🚀

---

## 🔑 Get a Free Groq API Key

Groq inference is **free** with generous rate limits:
1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Paste it in the sidebar or add to secrets

---

## 🏗️ Architecture

```
datalens-ai/
├── app.py              # Main Streamlit app
├── requirements.txt
├── .streamlit/
│   └── secrets.toml    # API keys (gitignored)
└── README.md
```

---

## 🧠 Tech Stack

- **Frontend** — Streamlit with custom CSS (Space Grotesk + JetBrains Mono)
- **Visualization** — Plotly (interactive), Matplotlib + Seaborn (static fallback)
- **AI Engine** — Groq Cloud with LLaMA-3.3-70B-Versatile
- **Data** — Pandas + NumPy

---

## 📸 Screenshots

> Add your own screenshots after deployment — `docs/screenshot_overview.png` etc.

---

## 📝 License

MIT — free to use, fork, and build on.

---

## 👩‍💻 Author

Built by **Shrey** · [LinkedIn](https://linkedin.com/in/shreyareddy) · [Medium](https://medium.com/@shreyareddy)
