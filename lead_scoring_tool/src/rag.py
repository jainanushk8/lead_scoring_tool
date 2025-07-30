import pandas as pd
import os
import sqlite3
import logging
import sys
import wikipediaapi
from transformers.pipelines import pipeline
import torch
import time
import re
import random # <--- ADD THIS IMPORT

# --- Logging Configuration (for rag.py when run independently) ---
# This basicConfig will only apply if rag.py is run directly.
# When imported by app.py, app.py's logging configuration will take precedence.
if not logging.getLogger().handlers:
    # Explicitly add a StreamHandler with UTF-8 encoding for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler.encoding = 'utf-8'  # Explicitly set encoding for the console handler # type: ignore
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO) # Set default level if not set by basicConfig

# --- Constants and Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Global Models/APIs ---
WIKI_TIMEOUT = 10 # seconds
WIKI_API_USER_AGENT = "SaaSquatchyLeadGenBot/1.0 (contact@yourdomain.com) (Wikipedia-API/0.6.0; https://github.com/martin-majlis/Wikipedia-API/)"

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent=WIKI_API_USER_AGENT,
    extract_format=wikipediaapi.ExtractFormat.WIKI, # Use WIKI for easier text processing
    timeout=WIKI_TIMEOUT # <--- Pass timeout directly in constructor
)
# No need to set wiki_wiki.timeout = WIKI_TIMEOUT here if passed in constructor
# No need to access wiki_wiki.user_agent here for logging if already passed
logging.info(f"Wikipedia: language={wiki_wiki.language}, user_agent: {WIKI_API_USER_AGENT}, extract_format={wiki_wiki.extract_format}") # Replaced direct access for logging
logging.info(f"Wikipedia API initialized with timeout.")


# Load summarization model
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device set to use {device}")
try:
    # Use a smaller model like 't5-small' for faster processing in a demo
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=0 if device == "cuda" else -1)
    logging.info("T5-small summarization pipeline loaded.")
except Exception as e:
    logging.error(f"Error loading T5-small model: {e}. Please ensure you have 'transformers' and 'torch' installed correctly.")
    summarizer = None # Set to None if loading fails

# --- Database Management ---
def connect_db(db_path):
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    logging.info(f"DB: Attempting to connect to database at {db_path} and ensure tables.")
    return conn

def create_table(conn):
    """Creates the 'insights' table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            industry TEXT UNIQUE,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Add a leads table to store enriched lead data, linking by id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY,
            company_name TEXT,
            industry TEXT,
            phone TEXT,
            address TEXT,
            email TEXT,
            revenue REAL,
            score REAL, -- Added score column
            insights TEXT,
            last_enriched DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    logging.info("DB: Database connection successful and tables checked/created.")

def get_cached_insight(conn, industry):
    """Retrieves a cached insight from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM insights WHERE industry = ?", (industry,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return None

def save_insight(conn, industry, summary):
    """Saves an insight to the database."""
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO insights (industry, summary) VALUES (?, ?)", (industry, summary))
    conn.commit()

def save_enriched_lead(conn, lead_data):
    """Saves an enriched lead to the 'leads' table."""
    cursor = conn.cursor()
    # Use INSERT OR REPLACE to update existing leads or insert new ones
    cursor.execute("""
        INSERT OR REPLACE INTO leads (id, company_name, industry, phone, address, email, revenue, score, insights, last_enriched)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        lead_data['id'],
        lead_data.get('company_name'),
        lead_data.get('industry'),
        lead_data.get('phone'),
        lead_data.get('address'),
        lead_data.get('email'),
        lead_data.get('revenue'),
        lead_data.get('score'), # Ensure 'score' is included
        lead_data.get('insights')
    ))
    conn.commit()

def load_enriched_leads_from_db(db_path) -> pd.DataFrame:
    """Loads enriched leads from the database into a DataFrame."""
    try:
        conn = sqlite3.connect(db_path) # Connect directly here as it's a standalone load
        cursor = conn.cursor()
        cursor.execute("SELECT id, company_name, industry, phone, address, email, revenue, score, insights FROM leads ORDER BY last_enriched DESC")
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        df = pd.DataFrame(rows, columns=columns)
        # Ensure 'id' is integer type for merging later
        if 'id' in df.columns:
            df['id'] = df['id'].astype(int)
        logging.info(f"DB: Loaded {len(df)} enriched leads from database.")
        return df
    except Exception as e:
        logging.error(f"DB Error: Could not load enriched leads from DB: {e}")
        return pd.DataFrame()


# --- RAG Logic ---
def get_industry_insight(industry: str, conn) -> str:
    """
    Retrieves and summarizes industry insights from Wikipedia, caching results.
    Improved search strategy: Try direct, then with "industry", then "sector".
    """
    # Clean industry name for better search queries
    clean_industry = re.sub(r'[^a-zA-Z0-9\s]', '', industry).strip()

    cached_insight = get_cached_insight(conn, clean_industry)
    if cached_insight:
        logging.info(f"Insight Fetcher: Found cached insight for '{clean_industry}'.")
        return cached_insight

    logging.info(f"Insight Fetcher: Fetching new insight for '{clean_industry}'.")
    page_content = ""
    search_queries = [
        clean_industry,
        f"{clean_industry} industry",
        f"{clean_industry} sector",
        f"{clean_industry} business"
    ]

    for query in search_queries:
        try:
            page = wiki_wiki.page(query)
            if page.exists() and page.text:
                page_content = page.text
                logging.info(f"Insight Fetcher: Found Wikipedia page for query '{query}'.")
                break
            else:
                logging.warning(f"Insight Fetcher: Wikipedia page for query '{query}' does not exist or is empty.")
        except Exception as e:
            logging.error(f"Insight Fetcher: Error fetching Wikipedia page for '{query}': {e}")
            page_content = "" # Reset content on error and try next query

    if not page_content:
        # Fallback if no specific page is found
        logging.warning(f"Insight Fetcher: Could not find relevant Wikipedia content for '{clean_industry}' after multiple attempts. Using generic message.")
        return f"Could not retrieve specific strategic insights for '{clean_industry}' from Wikipedia after multiple attempts. General overview might be limited or content not directly business-focused."

    # Limit content length for summarization for performance
    max_summary_input_length = 1000 # Characters, not tokens
    if len(page_content) > max_summary_input_length:
        page_content = page_content[:max_summary_input_length] + "..." # Truncate and add ellipsis

    if summarizer:
        try:
            # Ensure input is a string
            summary_text = summarizer(page_content, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
            # Clean up potential leading/trailing spaces or newlines from summarizer output
            summary_text = summary_text.strip()
            save_insight(conn, clean_industry, summary_text)
            logging.info(f"Insight Fetcher: Successfully summarized and cached insight for '{clean_industry}'.")
            return summary_text
        except Exception as e:
            logging.error(f"Insight Fetcher: Error during summarization for '{clean_industry}': {e}. Using raw text as fallback.", exc_info=True)
            # Fallback to a truncated version of the raw content if summarization fails
            fallback_summary = page_content.split('\n')[0][:200] + "..." if page_content else f"General information about {clean_industry} industry."
            save_insight(conn, clean_industry, fallback_summary) # Cache fallback
            return fallback_summary
    else:
        logging.warning("Summarizer not loaded. Skipping summarization and returning raw (truncated) Wikipedia content.")
        fallback_summary = page_content.split('\n')[0][:200] + "..." if page_content else f"General information about {clean_industry} industry."
        save_insight(conn, clean_industry, fallback_summary) # Cache fallback
        return fallback_summary

def enrich_with_rag(input_df: pd.DataFrame, output_csv: str, db_path: str) -> pd.DataFrame:
    """
    Enriches the DataFrame with industry insights using RAG.
    
    Args:
        input_df (pd.DataFrame): The DataFrame containing leads.
        output_csv (str): Path to save the enriched CSV.
        db_path (str): Path to the SQLite database for caching.
        
    Returns:
        pd.DataFrame: The DataFrame with an added 'insights' column.
    """
    logging.info("Attempting to enrich leads with RAG insights...")
    df = input_df.copy() # Work on a copy

    if df.empty:
        logging.warning("RAG Process: Input DataFrame is empty. No leads to enrich.")
        return df

    logging.info(f"RAG Process: Starting enrichment with {len(df)} leads.")

    # Ensure 'id' column exists for robust processing and database interaction
    if 'id' not in df.columns or df['id'].isnull().any():
        df['id'] = range(1, len(df) + 1)
        logging.info("Added 'id' column to DataFrame for robust processing.")

    # Ensure 'industry' column exists, fill NaNs if necessary
    if 'industry' not in df.columns:
        df['industry'] = 'Unknown'
        logging.warning("RAG Process: 'industry' column missing, defaulting to 'Unknown'.")
    else:
        df['industry'] = df['industry'].fillna('Unknown').astype(str)

    conn = None # Initialize conn to None
    try:
        conn = connect_db(db_path)
        create_table(conn)

        logging.info("🧠 Enriching industry insights using Wikipedia + T5-small (with improved search)...")
        # Get unique industries to minimize API calls
        unique_industries = df['industry'].unique()
        industry_insights = {}

        for industry in unique_industries:
            if industry == 'Unknown' or not industry.strip(): # Skip empty or unknown industries
                industry_insights[industry] = "No specific industry information available."
                continue
            logging.info(f"Insight Fetcher: Starting for industry '{industry}'.")
            insight = get_industry_insight(industry, conn)
            industry_insights[industry] = insight

        df['insights'] = df['industry'].map(industry_insights)
        logging.info("RAG Process: All industries processed for insights.")

        # Save the enriched DataFrame to CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        logging.info(f"RAG Process: Enriched data saved to CSV: {output_csv}")

        # Save enriched leads to the database
        try:
            # Select columns to save, only including 'score' if it exists in the DataFrame
            # Define all potential columns in the order they should appear in the DB.
            # Use a dictionary comprehension to ensure only existing columns are included.
            potential_columns = {
                'id': int, 'company_name': str, 'industry': str, 'phone': str,
                'address': str, 'email': str, 'revenue': float, 'score': float, 'insights': str
            }
            
            # Create a list of columns to select from the DataFrame that actually exist
            existing_columns_in_df = [col for col in potential_columns.keys() if col in df.columns]
            
            df_to_db = df[existing_columns_in_df].copy()

            # Ensure all expected columns for the DB table are present in df_to_db, filling missing ones
            # with appropriate default values before saving. This prevents issues if a column
            # (like 'score') is not in the source df, but is expected by the DB table.
            for col, dtype in potential_columns.items():
                if col not in df_to_db.columns:
                    if dtype == str:
                        df_to_db[col] = ''
                    elif dtype in [int, float]:
                        df_to_db[col] = 0 if dtype == int else 0.0
                    else:
                        df_to_db[col] = None
            
            # Reorder columns to match the DB table's expected order for consistency
            df_to_db = df_to_db[list(potential_columns.keys())]

            # Iterate through rows and save to DB
            for index, row in df_to_db.iterrows(): # <--- CHANGED: Use direct iteration
                save_enriched_lead(conn, row.to_dict())
            
            logging.info(f"DB: Saved {len(df)} enriched leads to database.")
        except Exception as e:
            logging.error(f"RAG Process Error: Failed to save enriched leads to DB: {e}", exc_info=True)


        logging.info(f"✅ RAG Enrichment complete for {len(df)} leads.")
        return df

    except Exception as e:
        logging.error(f"RAG Process Error: An unexpected error occurred during enrichment: {e}", exc_info=True)
        return input_df # Return original DataFrame if error occurs
    finally:
        if conn:
            conn.close()
            logging.info("DB: Database connection closed.")


# --- Run standalone for testing purposes ---
if __name__ == "__main__":
    logging.info("--- RAG Module Test Run ---")

    CSV_PATH = os.path.join(DATA_DIR, "leads.csv")
    DB_PATH = os.path.join(DATA_DIR, "leads.db")

    # Ensure leads.csv exists for testing, create mock if not
    if not os.path.exists(CSV_PATH):
        # Create a simple mock DataFrame for RAG testing
        mock_data = {
            'id': range(1, 11),
            'company_name': [f'Company {i}' for i in range(1, 11)],
            'industry': [
                'Real Estate', 'Technology', 'Construction', 'Healthcare', 'Retail',
                'Education', 'Hospitality', 'Finance', 'Manufacturing', 'Logistics'
            ],
            'phone': ['111-222-3333'] * 10,
            'address': ['123 Main St'] * 10,
            'email': [f'email{i}@example.com' for i in range(1, 11)],
            'revenue': [random.randint(500000, 5000000) for _ in range(10)], # random used here
            'score': [random.uniform(0.1, 0.9) for _ in range(10)] # random used here
        }
        mock_df = pd.DataFrame(mock_data)
        mock_df.to_csv(CSV_PATH, index=False)
        logging.info(f"Generated mock data for RAG testing and saved to {CSV_PATH}.")
    else:
        try:
            mock_df = pd.read_csv(CSV_PATH)
            logging.info(f"Loaded existing {len(mock_df)} leads from '{CSV_PATH}' for RAG testing.")
        except Exception as e:
            logging.error(f"Error loading existing CSV for RAG test: {e}. Generating new mock data.")
            # Fallback to generating if existing CSV is bad
            mock_data = {
                'id': range(1, 11),
                'company_name': [f'Company {i}' for i in range(1, 11)],
                'industry': [
                    'Real Estate', 'Technology', 'Construction', 'Healthcare', 'Retail',
                    'Education', 'Hospitality', 'Finance', 'Manufacturing', 'Logistics'
                ],
                'phone': ['111-222-3333'] * 10,
                'address': ['123 Main St'] * 10,
                'email': [f'email{i}@example.com' for i in range(1, 11)],
                'revenue': [random.randint(500000, 5000000) for _ in range(10)], # random used here
                'score': [random.uniform(0.1, 0.9) for _ in range(10)] # random used here
            }
            mock_df = pd.DataFrame(mock_data)
            mock_df.to_csv(CSV_PATH, index=False)
            logging.info(f"Generated new mock data for RAG testing and saved to {CSV_PATH}.")


    # Call enrich_with_rag with the DataFrame directly
    enriched_df_test = enrich_with_rag(input_df=mock_df, output_csv=CSV_PATH, db_path=DB_PATH)

    if enriched_df_test is not None:
        logging.info("\nFirst 5 leads after RAG enrichment (showing company, industry, and insights):")
        # Ensure 'insights' column is present before trying to display it
        display_cols = ['company_name', 'industry', 'insights']
        existing_display_cols = [col for col in display_cols if col in enriched_df_test.columns]
        logging.info(enriched_df_test[existing_display_cols].head().to_string())
    else:
        logging.warning("RAG enrichment test: `enrich_with_rag` returned None or an empty DataFrame.")
    logging.info("--- RAG Module Test Run Finished ---")