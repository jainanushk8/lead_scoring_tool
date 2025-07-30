import pandas as pd
import requests
import os
import random
from urllib.parse import urlparse # For extracting domain from URL

# Function to enrich leads with email
def enrich_leads(input_csv="../data/leads.csv", output_csv="../data/leads.csv"):
    df = pd.DataFrame()  # Ensure df is always defined

    try:
        df = pd.read_csv(input_csv)
        if df.empty:
            print("No leads to enrich")
            return df
        
        # --- Start of Refinement for Email Enrichment ---
        
        def generate_realistic_email(row):
            company_name = str(row["company_name"]).lower().replace(' ', '')
            website = str(row.get("website", "N/A")) # Safely get website column, default to N/A

            domain = None
            if website and website != "N/A" and "http" in website: # Check if it's a valid-looking URL
                try:
                    parsed_url = urlparse(website)
                    domain = parsed_url.netloc
                    if domain.startswith("www."):
                        domain = domain[4:] # Remove 'www.'
                except Exception:
                    domain = None
            
            # If a valid domain is extracted, use it
            if domain:
                common_prefixes = ["info", "contact", "sales", "support", "hello"]
                return f"{random.choice(common_prefixes)}@{domain}"
            else:
                # Fallback to company name if no valid website/domain found
                # Generate a few common patterns based on company name
                email_patterns = [
                    f"info@{company_name}.com",
                    f"contact@{company_name}.com",
                    f"sales@{company_name}.com",
                    f"support@{company_name}.com",
                    f"hello@{company_name}.com"
                ]
                return random.choice(email_patterns)
        
        df["email"] = df.apply(generate_realistic_email, axis=1)
        
        # --- End of Refinement ---
        
        df.to_csv(output_csv, index=False)
        print(f"Enriched leads saved to {output_csv}")
        return df

    except Exception as e:
        print(f"Error during enrichment: {e}")
        # If an error occurs, ensure the original df is returned, possibly with an "email" column of "N/A"
        if "email" not in df.columns:
            df["email"] = "N/A"
        df.to_csv(output_csv, index=False) # Attempt to save even on error
        print(f"📄 Data with potential errors saved to {output_csv}")
        return df

if __name__ == "__main__":
    # To test this, ensure your leads.csv has a 'website' column
    # The updated scrape.py should generate mock data with a 'website' column
    print("Attempting to enrich leads...")
    enriched_df = enrich_leads()
    if enriched_df is not None:
        print("\nFirst 5 leads after enrichment:")
        print(enriched_df[["company_name", "website", "email"]].head())