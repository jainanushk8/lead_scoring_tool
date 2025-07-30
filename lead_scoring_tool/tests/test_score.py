import sys
import pathlib

# Ensure src/ is on Python path
SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

import pandas as pd
from score import score_leads, generate_outreach_template

def test_score_leads_outputs_score_and_model_saved(tmp_path):
    df = pd.DataFrame([
        {"company_name":"TestCo","industry":"Technology","address":"123 New York","phone":"","email":"","revenue":1000000}
    ])
    test_csv = tmp_path / "leads.csv"
    df.to_csv(test_csv, index=False)

    scored = score_leads(input_csv=str(test_csv), output_csv=str(test_csv))
    assert "score" in scored.columns
    assert 0.0 <= scored["score"].iloc[0] <= 1.0

def test_generate_outreach_template_contains_company():
    lead = {"company_name": "TestCo", "industry": "Tech"}
    template = generate_outreach_template(lead)
    assert "TestCo" in template
    assert "Tech" in template
