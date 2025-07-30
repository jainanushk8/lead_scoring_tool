import streamlit as st
import pandas as pd
import os
import sys
import logging
import plotly.express as px
import sqlite3
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Logging Configuration (Place this at the very top of your app.py) ---
# This function ensures that the logging system is configured to output
# Unicode characters correctly to the console.
def configure_logging_for_unicode():
    # Remove any default handlers that might have been set by basicConfig
    # to avoid duplicate logs or conflicts.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Use basicConfig with the stream and encoding parameters directly.
    # This is the recommended and most robust way to set up the root logger
    # with a specific output stream and encoding.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,  # Direct logs to standard output
        encoding='utf-8',   # Specify UTF-8 encoding for the stream handler
        force=True          # Force re-configuration, useful in interactive environments like Streamlit
    )

# Call the configuration function as early as possible in your app.py
configure_logging_for_unicode()
# --- End Logging Configuration ---


# Import your custom modules
from src.scrape import scrape_yellow_pages
from src.score import score_leads
from src.rag import enrich_with_rag, load_enriched_leads_from_db

# --- Constants and Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Ensure data and models directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) 

DATA_PATH = os.path.join(DATA_DIR, 'leads.csv')
DB_PATH = os.path.join(DATA_DIR, 'leads.db')
MODEL_PATH = os.path.join(MODELS_DIR, 'lead_scorer.pkl')

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="SaaSquatchy Lead Intelligence Dashboard", page_icon="📈")

st.title("📈 SaaSquatchy Lead Intelligence Dashboard")
st.markdown("Automate lead generation, qualify prospects with AI scoring, and enrich data with real-time industry insights for personalized outreach.")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'last_scored_len' not in st.session_state:
    st.session_state.last_scored_len = 0
if 'rag_enriched' not in st.session_state:
    st.session_state.rag_enriched = False
if 'enriched_df' not in st.session_state:
    st.session_state.enriched_df = pd.DataFrame()

# --- Functions for Data Loading/Saving ---
def load_data():
    """Loads leads from CSV, scores them, and updates session state."""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, encoding='utf-8')
            if not df.empty:
                # Assign unique IDs if missing (important for merging)
                if 'id' not in df.columns:
                    df['id'] = range(1, len(df) + 1)
                    logging.info("Assigned 'id' column to loaded DataFrame.")

                # Check if new leads need scoring or if score column is missing
                # or if the number of leads has changed since last scoring
                if len(df) != st.session_state.last_scored_len or 'score' not in df.columns or df['score'].isnull().any():
                    with st.spinner("Scoring new leads..."):
                        # Pass DataFrame directly to score_leads, which now accepts a DataFrame
                        # and returns a scored DataFrame.
                        df = score_leads(df, DATA_PATH) # Pass df and data path for saving the scored data
                    st.session_state.last_scored_len = len(df)
                    df.to_csv(DATA_PATH, index=False, encoding='utf-8') # Save updated scores
                    logging.info(f"Scored {len(df)} leads and saved to CSV.")
                
                st.session_state.df = df.copy() # Store a copy in session state

                # Attempt to load enriched data from DB if exists and RAG was run
                if st.session_state.rag_enriched and os.path.exists(DB_PATH):
                    try:
                        enriched_db_df = load_enriched_leads_from_db(DB_PATH)
                        if not enriched_db_df.empty and 'id' in df.columns and 'id' in enriched_db_df.columns:
                            # Merge only the 'insights' column if 'id' matches
                            # Use `merge` and `fillna` to bring insights from DB into the main DataFrame
                            df_with_insights = pd.merge(df, enriched_db_df[['id', 'insights']], on='id', how='left', suffixes=('', '_db'))
                            # Prioritize insights from DB if they exist, otherwise keep original
                            df_with_insights['insights'] = df_with_insights['insights_db'].fillna(df_with_insights.get('insights', ''))
                            df_with_insights.drop(columns=['insights_db'], errors='ignore', inplace=True) # Drop the temporary column
                            st.session_state.enriched_df = df_with_insights.copy()
                            logging.info(f"Loaded and merged {len(st.session_state.enriched_df)} enriched leads from DB.")
                        else:
                            st.session_state.enriched_df = pd.DataFrame() # Clear if merge fails or no IDs
                            st.session_state.rag_enriched = False # Reset if loading fails
                    except Exception as e:
                        logging.error(f"Error loading enriched leads from DB: {e}")
                        st.error(f"Could not load enriched data from DB: {e}. Displaying basic data.")
                        st.session_state.rag_enriched = False # Reset if loading fails
                return
        except Exception as e:
            st.error(f"Error loading data from {DATA_PATH}: {e}. Please upload a valid CSV.")
            logging.error(f"Error loading data from CSV: {e}")
    st.session_state.df = pd.DataFrame() # Ensure df is empty if no file or error
    st.session_state.enriched_df = pd.DataFrame() # Ensure enriched_df is empty if no file or error
    st.session_state.rag_enriched = False


def save_data(df):
    """Saves DataFrame to CSV."""
    df.to_csv(DATA_PATH, index=False, encoding='utf-8')
    st.session_state.df = df.copy() # Store a copy
    st.session_state.last_scored_len = len(df)
    logging.info(f"Data saved to CSV: {DATA_PATH}")

# Initial load of data
load_data()

# --- Sidebar for Actions ---
st.sidebar.header("Actions")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Leads CSV", type="csv", help="Upload a CSV with your leads. It should contain at least 'company_name' and 'industry' columns for best results.")
if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file, encoding='utf-8')
        if 'id' not in new_df.columns:
            new_df['id'] = range(1, len(new_df) + 1) # Assign unique IDs if missing
        save_data(new_df)
        st.session_state.rag_enriched = False # Reset RAG status on new upload
        st.session_state.enriched_df = pd.DataFrame() # Clear enriched df
        st.success(f"Successfully uploaded and loaded {len(new_df)} leads!")
        st.rerun() # Rerun to trigger load_data and scoring
    except Exception as e:
        st.error(f"Error processing uploaded CSV: {e}. Please ensure it's a valid UTF-8 CSV.")
        logging.error(f"Error uploading CSV: {e}")

st.sidebar.markdown("---")

# Scrape New Leads
st.sidebar.subheader("Generate New Leads")
search_term = st.sidebar.text_input("Search Term (e.g., 'SaaS', 'Marketing Agency')", value="Software Development")
location = st.sidebar.text_input("Location (e.g., 'New York, NY', 'London')", value="Austin, TX")
max_pages = st.sidebar.slider("Max Pages to Scrape (Demo Limit)", 1, 5, 1)

if st.sidebar.button("Scrape Yellow Pages"):
    if search_term and location:
        with st.spinner(f"Scraping '{search_term}' in '{location}' for up to {max_pages} page(s)... This may take a while."):
            try:
                scraped_df = scrape_yellow_pages(search_term, location, max_pages)
                if not scraped_df.empty:
                    # Append new leads to existing, handling duplicates and IDs
                    if 'id' not in scraped_df.columns:
                        max_id = st.session_state.df['id'].max() if 'id' in st.session_state.df.columns and not st.session_state.df.empty else 0
                        scraped_df['id'] = range(max_id + 1, max_id + 1 + len(scraped_df))

                    combined_df = pd.concat([st.session_state.df, scraped_df], ignore_index=True)
                    # Deduplicate based on a combination of columns to avoid re-scraping same exact entries
                    combined_df.drop_duplicates(subset=['company_name', 'phone', 'address'], inplace=True, keep='first')

                    save_data(combined_df)
                    st.session_state.rag_enriched = False # Reset RAG status on new scrape
                    st.session_state.enriched_df = pd.DataFrame() # Clear enriched df
                    st.success(f"Scraped {len(scraped_df)} new leads. Total leads: {len(st.session_state.df)}")
                    st.rerun() # Rerun to trigger load_data and scoring
                else:
                    st.info("No new leads found for the given criteria.")
            except Exception as e:
                st.error(f"Error during scraping: {e}. Please try again later or check your inputs.")
                logging.error(f"Scraping error: {e}")
    else:
        st.sidebar.warning("Please enter both a search term and a location.")

st.sidebar.markdown("---")

# RAG Enrichment
if st.sidebar.button("Enrich with Industry Insights (RAG)", help="Uses AI to find and summarize industry insights for each lead. This can take some time."):
    if not st.session_state.df.empty:
        with st.spinner("Retrieving and generating industry insights... This might take a while for many leads."):
            try:
                # Pass the current DataFrame from session state to enrich_with_rag
                st.session_state.enriched_df = enrich_with_rag(
                    input_df=st.session_state.df, # Pass the DataFrame directly
                    output_csv=DATA_PATH,
                    db_path=DB_PATH
                )
                st.session_state.rag_enriched = True
                st.success("🎉 Leads successfully enriched with industry insights!")
            except Exception as e:
                st.error(f"Error during RAG enrichment: {e}. Check logs for details.")
                logging.error(f"RAG enrichment error: {e}")
                st.session_state.rag_enriched = False # Reset if error
    else:
        st.sidebar.warning("Please upload or scrape leads first to enrich them.")

st.sidebar.markdown("---")

# --- Main Content Area ---
if st.session_state.df.empty:
    st.info("Upload leads via CSV or scrape new leads from Yellow Pages using the sidebar to get started!")
else:
    st.header("📊 Lead Dashboard")

    # Determine which DataFrame to display and filter
    # If RAG has been run and enriched_df is not empty, use it. Otherwise, use the base df.
    current_display_df = st.session_state.enriched_df if st.session_state.rag_enriched and not st.session_state.enriched_df.empty else st.session_state.df

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total Leads", value=len(current_display_df))

    # Filtering by Score
    with col2:
        min_score = 0.0 # Initialize min_score to avoid unbound error
        if 'score' in current_display_df.columns and not current_display_df['score'].isnull().all():
            min_score = st.slider("Minimum Score", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                  help="Filter leads by their predicted 'score'. Higher scores indicate a better fit.")
            filtered_df = current_display_df[current_display_df['score'] >= min_score].copy()
            st.metric(label="Qualified Leads", value=len(filtered_df))
        else:
            filtered_df = current_display_df.copy() # No score column, so no score filter
            st.warning("No 'score' column found in data to filter by.")
            st.metric(label="Qualified Leads", value=len(filtered_df)) # Still show total leads as qualified

    with col3:
        if 'industry' in current_display_df.columns:
            num_industries = current_display_df['industry'].nunique()
            st.metric(label="Unique Industries", value=num_industries)
        else:
            st.warning("No 'industry' column found.")
            st.metric(label="Unique Industries", value=0)

    st.markdown("---")

    # --- Visualizations ---
    st.subheader("Visualizations")
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        if 'score' in current_display_df.columns and not current_display_df['score'].isnull().all():
            st.subheader("Lead Score Distribution")
            fig_score = px.histogram(current_display_df, x='score', nbins=20, title='Distribution of Lead Scores')
            st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.info("No 'score' column available for visualization.")

    with vis_col2:
        if 'industry' in current_display_df.columns and not current_display_df['industry'].isnull().all():
            st.subheader("Leads per Industry")
            industry_counts = current_display_df['industry'].value_counts().reset_index()
            industry_counts.columns = ['Industry', 'Count']
            fig_industry = px.bar(industry_counts, x='Industry', y='Count', title='Number of Leads per Industry')
            st.plotly_chart(fig_industry, use_container_width=True)
        else:
            st.info("No 'industry' column available for visualization.")

    st.markdown("---")

    # --- Leads Table Display ---
    st.subheader("Leads Table")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Data Export ---
    st.subheader("Export Data")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_export = filtered_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="Download Filtered Leads (CSV)",
            data=csv_export,
            file_name=f"qualified_leads_{datetime.date.today()}.csv",
            mime="text/csv",
            help="Download the currently filtered leads as a CSV file."
        )
    with col_dl2:
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                db_bytes = f.read()
            st.download_button(
                label="Download SQLite Database",
                data=db_bytes,
                file_name=f"leads_db_{datetime.date.today()}.db",
                mime="application/octet-stream", # Standard MIME type for arbitrary binary data
                help="Download the SQLite database containing all enriched leads and cached insights."
            )
        else:
            st.info("SQLite Database not found. Run RAG enrichment to create it.")

    st.markdown("---")

    # --- Outreach Email Generator ---
    st.subheader("📧 Outreach Email Generator")

    # Use the most complete DF for email generation
    source_df_for_email = st.session_state.enriched_df if st.session_state.rag_enriched and not st.session_state.enriched_df.empty else st.session_state.df

    if not source_df_for_email.empty:
        # Create a display column for selection that includes company name and score
        if 'score' in source_df_for_email.columns:
            display_options = source_df_for_email.apply(lambda row: f"{row['company_name']} (Score: {row['score']:.2f})" if pd.notna(row['score']) else f"{row['company_name']} (Score: N/A)", axis=1).tolist()
        else:
            display_options = source_df_for_email.apply(lambda row: f"{row['company_name']} ({row['industry']})", axis=1).tolist()
            
        selected_lead_display = st.selectbox("Select a Lead to Generate Email:", display_options)

        if selected_lead_display:
            # Find the actual row based on the selected display option
            if 'score' in source_df_for_email.columns:
                 # Extract company name from the display string
                company_name_from_display = selected_lead_display.split(" (Score:")[0].strip()
                selected_lead_row = source_df_for_email[source_df_for_email['company_name'] == company_name_from_display].iloc[0]
            else: # If no score column, extract from the industry format
                company_name_from_display = selected_lead_display.split(" (")[0].strip()
                selected_lead_row = source_df_for_email[source_df_for_email['company_name'] == company_name_from_display].iloc[0]


            company_name = selected_lead_row.get('company_name', 'Valued Prospect')
            contact_person = selected_lead_row.get('contact_person', 'Sir/Madam')
            industry = selected_lead_row.get('industry', 'their industry')
            revenue = selected_lead_row.get('revenue', 'their estimated revenue')
            insights = selected_lead_row.get('insights', 'no specific industry insights available yet.') # This is from RAG!

            email_subject = f"Partnership Opportunity with {company_name} in the {industry} Sector"
            email_body = f"""Dear {contact_person},

I hope this email finds you well.

I'm writing to you from SaaSquatchy Leads. We specialize in helping companies like yours, particularly in the {industry} sector, optimize their sales and lead generation efforts.

We've observed that {company_name} is a key player in {industry}. Based on our analysis, we noted {insights}. This presents a unique opportunity for synergy.

Given {company_name}'s {revenue if pd.notna(revenue) else 'size/potential'}, we believe our [Your Product/Service] could significantly benefit your operations by [mention a specific benefit related to their industry or insights, e.g., "streamlining your lead qualification process," "identifying high-value customers"].

Would you be open to a brief 15-minute call next week to explore how we can help you achieve [specific goal, e.g., "accelerate growth" or "increase market share"]?

Best regards,

[Your Name/SaaSquatchy Sales Team]
[Your Title]
[Your Company]
[Your Website/Contact Info]
"""
            st.text_input("Email Subject:", value=email_subject)
            st.text_area("Email Body:", value=email_body, height=300)

            st.markdown(
                """
                **💡 Tip:** Copy the subject and body, then paste into your email client. Remember to personalize further!
                """
            )
    else:
        st.info("No leads available to generate an email. Please upload or scrape leads first.")