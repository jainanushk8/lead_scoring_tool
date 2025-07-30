import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import random
import logging
import sys

# --- Logging Configuration (for score.py when run independently) ---
# This basicConfig will only apply if score.py is run directly.
# When imported by app.py, app.py's logging configuration will take precedence.
# We explicitly set up a stream handler here with encoding for direct execution clarity.
if not logging.getLogger().handlers:
    # Use basicConfig with stream and encoding parameters
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("score_log.log"),
                            # Explicitly set encoding for the StreamHandler
                            logging.StreamHandler(sys.stdout) # Initial stream handler
                        ],
                        encoding='utf-8' # <-- ADD THIS FOR THE FILE HANDLER AND DEFAULT STREAM HANDLER
                        )
    # The above basicConfig might create a default StreamHandler.
    # To ensure sys.stdout handler is UTF-8 for standalone script, let's explicitly add it if not present.
    # This might be redundant with the `encoding` arg above, but ensures the stdout stream handler
    # is explicitly UTF-8 in case of complex logging setups.
    # A cleaner approach for standalone scripts to directly control stdout:
    # Remove default handlers created by basicConfig if they're not UTF-8.
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr: # basicConfig often uses stderr by default for console
            logging.getLogger().removeHandler(handler)
    
    # Add a new StreamHandler for sys.stdout with explicit UTF-8 encoding
    # This is the most reliable way to ensure console output is UTF-8 for standalone scripts.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

# Define the path where the trained model will be saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "lead_scorer.pkl")

# Function to load or train the model
def load_or_train_model(model_path):
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logging.info(f"ML Model: Loaded existing model from {model_path}")
            return model
        except Exception as e:
            logging.warning(f"ML Model: Could not load model from {model_path}: {e}. Training a new dummy model.")
    else:
        logging.info("ML Model: No existing model found. Training a new dummy model.")

    # --- Dummy Model Training Data Generation ---
    # This dummy data should cover your target industries and locations to train the OHE
    dummy_data = {
        "company_name": [
            "AI Solutions Inc.", "MediCorp Global", "Local Cafe Co.", "Fashion Retailers",
            "Quantum Innovations", "Investment Pros LLC", "BioPharm Research",
            "FutureTech Corp", "Urban Retail Outlet", "Global FinTech Hub"
        ],
        "industry": [
            "Technology", "Healthcare", "Food & Beverage", "Retail",
            "AI", "Finance", "Biotech", "Software", "Retail", "Fintech"
        ],
        "address": [
            "123 Silicon Valley, San Francisco", "456 Main St, New York", "789 Elm St, Los Angeles",
            "101 Broadway, Chicago", "404 Innovation Dr, Bangalore", "888 Capital Rd, Houston",
            "99 Bio Blvd, San Francisco", "500 Tech Ave, London", "202 Market St, New York",
            "777 Crypto Rd, Singapore"
        ],
        "revenue": [
            2500000, 1800000, 600000, 300000, 3000000, 1000000, 2200000,
            2800000, 450000, 3500000
        ]
    }
    dummy_df = pd.DataFrame(dummy_data)

    # Apply the same target labeling logic as in score_leads
    target_industries = ["technology", "software", "ai", "data science", "healthcare", "biotech", "finance", "fintech"]
    target_locations = ["new york", "san francisco", "london", "bangalore", "singapore", "los angeles", "chicago"]
    REVENUE_HIGH_THRESHOLD = 2000000
    REVENUE_MEDIUM_THRESHOLD = 500000

    y_dummy = []
    for index, row in dummy_df.iterrows():
        industry_match = any(t in str(row["industry"]).lower() for t in target_industries)
        location_match = any(t in str(row["address"]).lower() for t in target_locations)
        is_high_revenue = row["revenue"] >= REVENUE_HIGH_THRESHOLD
        is_medium_revenue = row["revenue"] >= REVENUE_MEDIUM_THRESHOLD

        if (industry_match and is_high_revenue) or \
           (location_match and is_high_revenue) or \
           (industry_match and location_match and is_medium_revenue):
            y_dummy.append(1)
        else:
            y_dummy.append(0)

    X_dummy = dummy_df[["industry", "address", "revenue"]]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["industry", "address"]),
        ("num", "passthrough", ["revenue"])
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    model_pipeline.fit(X_dummy, y_dummy)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_pipeline, model_path)
    logging.info(f"ML Model: Trained and saved new dummy model to {model_path}")
    return model_pipeline

# Load the model when the module is imported
# This ensures the model is ready when score_leads is called by app.py
lead_scorer_model = load_or_train_model(MODEL_PATH)


# --- Function to score leads (Updated signature to accept DataFrame directly) ---
def score_leads(input_df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    """
    Scores leads based on industry, location, and revenue, and saves the results.
    
    Args:
        input_df (pd.DataFrame): The DataFrame containing leads to be scored.
        output_csv (str): The path to the CSV file where the scored leads will be saved.
    
    Returns:
        pd.DataFrame: The DataFrame with an added 'score' column, sorted by score.
    """
    logging.info(f"Scoring: Starting lead scoring for {len(input_df)} leads.")
    df = input_df.copy() # Work on a copy to avoid modifying the original DataFrame

    if df.empty:
        logging.warning("Scoring: Input DataFrame is empty. No leads to score.")
        return df

    # Ensure 'revenue' column exists and is numeric
    if 'revenue' not in df.columns or not pd.api.types.is_numeric_dtype(df['revenue']):
        logging.warning("Scoring: 'revenue' column missing or not numeric. Generating mock revenue.")
        df['revenue'] = [random.randint(200000, 5000000) for _ in range(len(df))]
    
    # Ensure 'industry' and 'address' columns are treated as strings
    df['industry'] = df['industry'].astype(str).fillna('') # Fill NaN with empty string
    df['address'] = df['address'].astype(str).fillna('') # Fill NaN with empty string


    # --- Automated "Training" (Labeling) for Lead Scoring ---
    # This section dynamically labels leads based on defined criteria for training the model.
    # In a real-world scenario, you would use actual conversion data (historical labels)
    # to train your model, not these rule-based labels for production.
    # However, for a demo, this provides a dynamic way to "train" the model.

    target_industries = ["technology", "software", "ai", "data science", "healthcare", "biotech", "finance", "fintech"]
    target_locations = ["new york", "san francisco", "london", "bangalore", "singapore", "los angeles", "chicago"]
    
    REVENUE_HIGH_THRESHOLD = 2000000
    REVENUE_MEDIUM_THRESHOLD = 500000

    y_labels = [] # This will be our target variable 'y'
    for index, row in df.iterrows():
        industry_match = any(t in row["industry"].lower() for t in target_industries)
        location_match = any(t in row["address"].lower() for t in target_locations)
        is_high_revenue = row["revenue"] >= REVENUE_HIGH_THRESHOLD
        is_medium_revenue = row["revenue"] >= REVENUE_MEDIUM_THRESHOLD

        if (industry_match and is_high_revenue) or \
           (location_match and is_high_revenue) or \
           (industry_match and location_match and is_medium_revenue):
            y_labels.append(1) # High potential lead
        else:
            y_labels.append(0) # Lower potential lead
    
    # --- Feature Selection for Prediction ---
    X_predict = df[["industry", "address", "revenue"]] 

    if lead_scorer_model:
        try:
            # Fit the preprocessor and classifier (retraining the model each time)
            # This approach re-fits the model with the current DataFrame.
            # In a real app, you'd load a *pre-trained* model and only use .predict_proba
            # on new data. The current setup retrains for demonstration flexibility.
            
            # Re-fit the model with the current data and dynamically generated labels
            lead_scorer_model.fit(X_predict, y_labels) # Use y_labels for re-fitting
            
            # Predict probabilities
            probas = lead_scorer_model.predict_proba(X_predict)
            
            # Assign the score to the DataFrame
            if lead_scorer_model.classes_.shape[0] > 1:
                positive_class_idx = list(lead_scorer_model.classes_).index(1)
                df["score"] = probas[:, positive_class_idx]
            else:
                # Fallback if only one class was generated in y_labels (less likely with balanced data)
                df["score"] = probas[:, 0] if lead_scorer_model.classes_[0] == 1 else 0.0
            
            logging.info("Scoring: Leads scored successfully.")

        except Exception as e:
            logging.error(f"Scoring Error: Failed to predict scores: {e}. Setting score to 0.0.", exc_info=True)
            df['score'] = 0.0 # Assign 0.0 if prediction fails
    else:
        logging.error("Scoring Error: Lead scorer model not loaded. Cannot score leads. Setting score to 0.0.")
        df['score'] = 0.0

    # Sort leads by score in descending order
    df = df.sort_values("score", ascending=False)
    
    # Save the scored DataFrame to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logging.info(f"Scoring: Scored leads saved to CSV: {output_csv}")

    # Save the (re-trained) model
    joblib.dump(lead_scorer_model, MODEL_PATH)
    logging.info(f"Scoring: Model re-trained and saved to {MODEL_PATH}")

    logging.info(f"✅ Scored {len(df)} leads.") # This line contains the emoji
    return df

# Outreach email template
def generate_outreach_template(lead_row: pd.Series) -> str:
    """Generates a personalized outreach email template for a given lead."""
    # Ensure all accesses use .get() for robustness in case a column is missing
    company_name = lead_row.get('company_name', 'Valued Prospect')
    contact_person = lead_row.get('contact_person', 'Sir/Madam') # Assuming 'contact_person' might exist
    industry = lead_row.get('industry', 'their industry')
    revenue = lead_row.get('revenue', 'their estimated revenue')
    insights = lead_row.get('insights', 'no specific industry insights available yet.') # From RAG!
    score = lead_row.get('score', 0.0)

    # Make revenue display more readable
    if pd.notna(revenue) and isinstance(revenue, (int, float)):
        revenue_display = f"${revenue:,.0f}"
    else:
        revenue_display = "their estimated revenue"

    # Craft a more compelling and Caprae Capital-aligned email template
    return f"""Subject: Partnership Opportunity with {company_name} - High Potential Lead

Dear {contact_person},

We at Caprae Capital specialize in transforming businesses through strategic initiatives and AI leverage. Your company, operating in the {industry} sector, stands out as a high-potential opportunity for our M&A as a Service model.

**Key Insight:** {insights}

We believe our unique approach, which includes introducing practical AI solutions and operational streamlining, can help {company_name} unlock new growth opportunities and achieve significant value creation.

Given {company_name}'s revenue of {revenue_display}, we see significant potential for synergistic growth.

We would be keen to explore how a partnership with Caprae Capital could lead to a successful, long-term journey for {company_name}.

Please let us know your availability for a brief introductory call.

Best regards,

[Your Name]
[Your Title]
Caprae Capital
[Your Contact Information]
"""

# Run standalone for testing purposes
if __name__ == "__main__":
    logging.info("--- Score Module Test Run ---")

    # Ensure some data exists for scoring. If leads.csv is not present, create mock data.
    test_csv_path = os.path.join(BASE_DIR, "..", "data", "test_leads_for_scoring.csv")
    if not os.path.exists(test_csv_path):
        logging.info("test_leads_for_scoring.csv not found, generating mock data for scoring test...")
        # Create a simple mock DataFrame that covers different scenarios for the new scoring logic
        mock_leads = pd.DataFrame([
            {"company_name": "AI Solutions Inc.", "industry": "Technology", "address": "123 Silicon Valley, San Francisco", "revenue": 2500000}, # Target Industry, Target Loc, High Revenue -> High Score (1)
            {"company_name": "MediCorp Global", "industry": "Healthcare", "address": "456 Main St, New York", "revenue": 1800000}, # Target Industry, Target Loc, Medium Revenue -> High Score (1)
            {"company_name": "Local Cafe Co.", "industry": "Food & Beverage", "address": "789 Elm St, Los Angeles", "revenue": 600000}, # Non-Target Industry, Target Loc, Medium Revenue -> Medium/Low Score (0)
            {"company_name": "Fashion Retailers", "industry": "Retail", "address": "101 Broadway, Chicago", "revenue": 300000}, # Non-Target, Non-Target Loc, Low Revenue -> Low Score (0)
            {"company_name": "Quantum Innovations", "industry": "AI", "address": "404 Innovation Dr, Bangalore", "revenue": 3000000}, # Target Industry, Target Loc, High Revenue -> High Score (1)
            {"company_name": "Investment Pros LLC", "industry": "Finance", "address": "888 Capital Rd, Houston", "revenue": 1000000}, # Target Industry, Non-Target Loc, Medium Revenue -> Low Score (0)
            {"company_name": "BioPharm Research", "industry": "Biotech", "address": "99 Bio Blvd, San Francisco", "revenue": 2200000} # Target Industry, Target Loc, High Revenue -> High Score (1)
        ])
        os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)
        mock_leads.to_csv(test_csv_path, index=False)
        logging.info("Mock data generated for testing.")
    else:
        mock_leads = pd.read_csv(test_csv_path)
        logging.info(f"Loaded existing {len(mock_leads)} leads from '{test_csv_path}' for scoring testing.")

    # Call score_leads with the DataFrame directly
    scored_df = score_leads(input_df=mock_leads, output_csv=test_csv_path)

    if scored_df is not None:
        logging.info("\nFirst 10 leads after scoring (sorted by score):")
        logging.info(scored_df[["company_name", "industry", "address", "revenue", "score"]].head(10).to_string())

        # Example of generating an outreach template for the top-scoring lead
        if not scored_df.empty:
            top_lead = scored_df.iloc[0] # Using .iloc[0] for first row, directly a Series
            # Simulate adding an 'insights' field, as the RAG step would do this in the full pipeline
            # If your test_leads_for_scoring.csv already has an 'insights' column, this will be used.
            if 'insights' not in top_lead:
                top_lead['insights'] = "The technology sector is experiencing rapid growth in AI integration, making companies like this prime candidates for strategic investment in an M&A context."
            logging.info("\nGenerated outreach template for top lead:")
            logging.info(generate_outreach_template(top_lead))
    else:
        logging.warning("Scoring test: `score_leads` returned None or an empty DataFrame.")
    logging.info("--- Score Module Test Run Finished ---")