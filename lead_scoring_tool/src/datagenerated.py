import random
import csv
from faker import Faker

fake = Faker()

industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Education', 'Real Estate', 'Hospitality', 'Construction']
cities = ['New York', 'Chicago', 'San Francisco', 'Los Angeles', 'Houston', 'Miami', 'Seattle', 'Boston']
street_names = ['Main St', 'Oak Ave', 'Pine Rd', 'Market St', 'Broadway', '5th Ave', 'Sunset Blvd', 'Lakeshore Dr']

def generate_company_name(industry):
    suffixes = ['Corp', 'LLC', 'Solutions', 'Group', 'Inc', 'Co', 'Systems', 'Partners']
    return f"{fake.company().split()[0]} {random.choice(suffixes)}"

def generate_phone():
    return f"{random.randint(200, 999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def generate_address():
    return f"{random.randint(100, 9999)} {random.choice(street_names)}"

def generate_email(company_name):
    domain = fake.free_email_domain()
    company_slug = company_name.lower().replace(' ', '')
    return f"info@{company_slug}.{domain.split('.')[-2]}.com"

def generate_revenue():
    return random.randint(100000, 10000000)

# Write to CSV
with open('synthetic_yellowpages_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['company_name', 'industry', 'phone', 'address', 'city', 'email', 'revenue']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _ in range(200):
        industry = random.choice(industries)
        company_name = generate_company_name(industry)
        city = random.choice(cities)
        address = generate_address()
        phone = generate_phone()
        email = generate_email(company_name)
        revenue = generate_revenue()

        writer.writerow({
            'company_name': company_name,
            'industry': industry,
            'phone': phone,
            'address': address,
            'city': city,
            'email': email,
            'revenue': revenue
        })

print("✅ 200 rows of synthetic Yellow Pages-style business data saved to 'synthetic_yellowpages_data.csv'.")
