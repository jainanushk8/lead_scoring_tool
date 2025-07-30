AI-Readiness Pre-Screening Challenge Report
Overview
This project enhances SaaSquatch’s lead generation for acquisition entrepreneurs by integrating lead scoring, enrichment, RAG-based insights, and outreach templating. Built in 5 hours, it aligns with Caprae Capital’s mission to drive AI-powered business transformation.
Approach

Data Source: Scraped 50-100 leads from Yellow Pages using requests and BeautifulSoup.
Enrichment: Simulated email lookup with Hunter.io (free API placeholder).
Scoring: Trained a scikit-learn RandomForestClassifier to predict lead quality based on industry, location, and mock revenue.
RAG: Used Wikipedia API and DistilBERT to fetch and summarize industry insights.
UI: Streamlit web app for filtering, viewing, and exporting leads.

Model Selection

Libraries: requests, BeautifulSoup, pandas, streamlit, scikit-learn, transformers, wikipedia-api (all free).
ML Model: RandomForestClassifier for predictive scoring, trained on synthetic labels.
RAG: Wikipedia for retrieval, DistilBERT for summarization.
Future: Integrate Grok 3 for advanced email generation or RAG for multi-source fetching.

Performance Evaluation

Business Use Case: Prioritizes high-potential leads, streamlining acquisition workflows.
UX/UI: Streamlit interface with sliders and tables for ease of use.
Technicality: Efficient scraping, ML scoring, and RAG with error handling.
Design: Clean, professional Streamlit layout.
Other: Creative RAG insights and ML scoring enhance lead quality.

Business Value
The tool saves time by prioritizing high-value leads and providing actionable insights via RAG, supporting Caprae’s AI-driven acquisition strategy.
Word Count: 250