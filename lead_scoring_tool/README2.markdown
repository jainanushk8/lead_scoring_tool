# AI-Powered Lead Generation Tool

This project enhances SaaSquatch with lead scoring, enrichment, RAG insights, and outreach templating for acquisition entrepreneurs.

## Setup Instructions
1. **Clone Repository**:
   ```bash
   git clone <repository_url>
   cd lead_scoring_tool
   ```
2. **Install Dependencies** (Windows):
   ```bash
   pip install requests beautifulsoup4 pandas streamlit scikit-learn transformers wikipedia-api
   ```
3. **Run Scripts**:
   - Scrape: `python src/scrape.py`
   - Enrich: `python src/enrich.py`
   - Score: `python src/score.py`
   - RAG: `python src/rag.py`
   - App: `streamlit run src/app.py`
4. **View Results**: Open the Streamlit app in your browser.

## Requirements
- Python 3.8+
- Windows (VS Code)
- Free APIs: Hunter.io (simulated), Wikipedia

## Files
- `src/scrape.py`: Scrapes leads from Yellow Pages.
- `src/enrich.py`: Enriches with emails.
- `src/score.py`: ML-based scoring with scikit-learn.
- `src/rag.py`: RAG for industry insights.
- `src/app.py`: Streamlit app.
- `docs/report.md`: Project report.
- `demo.ipynb`: Optional demo.

## Notes
- Data in `data/leads.csv`.
- Comply with Yellow Pages’ terms.
- Video walkthrough provided separately.