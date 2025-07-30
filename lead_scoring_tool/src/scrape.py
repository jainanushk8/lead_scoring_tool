import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random

# Define a list of common User-Agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 13_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/83.0.4103.88 Mobile/15E148 Safari/604.1"
]

DATA_PATH = os.path.join("..", "data", "leads.csv")

def get_random_user_agent():
    """Returns a random User-Agent from the predefined list."""
    return random.choice(USER_AGENTS)

def scrape_yellow_pages(search_term="business", location="New York", max_pages=3, results_per_page=30):
    all_leads = []
    
    # Loop through multiple pages
    for page in range(max_pages):
        # Yellow Pages pagination uses 'page' parameter, starting from 1
        # Each page seems to show 30 results by default, page=1 corresponds to start=0, page=2 to start=30, etc.
        # It's better to verify this on the actual Yellow Pages website for accuracy.
        # For this example, let's assume 'page' increments by 1, and the site automatically handles results per page.
        # If Yellow Pages uses 'start' parameter (e.g., ?start=0, ?start=30), we would adjust the URL.
        # For Yellow Pages, it seems to be just `page=N` in the URL, or no `page` for the first.
        
        # Construct URL with pagination
        # Example for Yellow Pages: https://www.yellowpages.com/search?search_terms=restaurants&geo_location_terms=new+york&page=2
        # For the first page, `page` parameter might be omitted or set to 1.
        current_url = f"https://www.yellowpages.com/search?search_terms={search_term}&geo_location_terms={location}"
        if page > 0: # For subsequent pages
            current_url += f"&page={page + 1}" 
            
        print(f"Attempting to scrape page {page + 1} from: {current_url}")

        headers = {
            "User-Agent": get_random_user_agent() # Use a random User-Agent for each request
        }

        try:
            response = requests.get(current_url, headers=headers, timeout=15) # Increased timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.text, "html.parser")

            results = soup.find_all("div", class_="result")

            if not results:
                print(f"No results found on page {page + 1}. Ending scraping.")
                break # No more results, stop scraping

            for result in results:
                name_tag = result.find("a", class_="business-name")
                industry_tag = result.find("div", class_="categories")
                phone_tag = result.find("div", class_="phones")
                address_tag = result.find("div", class_="street-address")
                
                # Extract website if available (Yellow Pages often has this on business detail pages or as a link)
                # For quick scraping, let's look for a direct link within the main search result div
                website_tag = result.find("a", class_="website-link") # Common class for website links

                leads_dict = {
                    "company_name": name_tag.text.strip() if name_tag else "N/A",
                    "industry": industry_tag.text.strip() if industry_tag else "N/A",
                    "phone": phone_tag.text.strip() if phone_tag else "N/A",
                    "address": address_tag.text.strip() if address_tag else "N/A",
                    "website": website_tag['href'] if website_tag and website_tag.has_attr('href') else "N/A", # Extract href attribute
                    "email": "N/A",  # Email not typically directly available on Yellow Pages search results
                    "revenue": random.randint(500000, 5000000) # Randomize mock revenue for realism
                }
                all_leads.append(leads_dict)
            
            print(f"Scraped {len(results)} leads from page {page + 1}. Total leads: {len(all_leads)}")

            # Add a random delay before the next request
            sleep_time = random.uniform(2, 5) # Random delay between 2 and 5 seconds
            print(f"Waiting for {sleep_time:.2f} seconds before next page...")
            time.sleep(sleep_time)

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request failed on page {page + 1}: {e}")
            print("Trying next page (if any) or ending scraping...")
            # Optionally add a longer backoff here if a 429 (Too Many Requests) is hit
        except Exception as e:
            print(f"⚠️ Error parsing page {page + 1}: {e}")
            print("Trying next page (if any) or ending scraping...")

    if all_leads:
        df = pd.DataFrame(all_leads)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"✅ Scraped {len(df)} total leads and saved to {DATA_PATH}")
        return df
    else:
        print("⚠️ No leads scraped. Using mock data instead...")
        mock_data = [
            {"company_name": "Tech Innovators", "industry": "Technology", "phone": "123-456-7890", "address": "123 Silicon Valley, CA", "website": "techinnovators.com", "email": "info@techinnovators.com", "revenue": 1500000},
            {"company_name": "MediCare Plus", "industry": "Healthcare", "phone": "987-654-3210", "address": "456 Wellness Blvd, TX", "website": "medicareplus.org", "email": "contact@medicareplus.com", "revenue": 750000},
            {"company_name": "Global Finance Corp", "industry": "Finance", "phone": "555-123-4567", "address": "789 Wall St, NY", "website": "globalfinance.net", "email": "hello@globalfinance.com", "revenue": 2000000},
            {"company_name": "Urban Boutiques", "industry": "Retail", "phone": "222-333-4444", "address": "101 Fashion Row, CA", "website": "urbanboutiques.shop", "email": "support@urbanboutiques.com", "revenue": 400000}
        ]
        df = pd.DataFrame(mock_data)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"📄 Mock data saved to {DATA_PATH}")
        return df


if __name__ == "__main__":
    # Example usage: scrape 'restaurants' in 'Los Angeles' for up to 2 pages
    scraped_df = scrape_yellow_pages(search_term="restaurants", location="Los Angeles", max_pages=2)
    print("\nFirst 5 leads from the scraped data:")
    print(scraped_df.head())

    # You can also test with the default values
    # scraped_df = scrape_yellow_pages()