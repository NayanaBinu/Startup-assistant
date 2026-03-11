import json

def load_schemes(filename):
    """Load processed schemes from JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def match_schemes(schemes, domain=None, registration=None, stage=None):
    """Filter schemes based on given eligibility."""
    results = []
    for scheme in schemes:
        eligible = True

        if domain and domain != "any" and domain not in scheme["eligibility"]["domain"]:
            eligible = False
        if registration and registration != "any" and registration not in scheme["eligibility"]["registration"]:
            eligible = False
        if stage and stage != "any" and stage not in scheme["eligibility"]["stage"]:
            eligible = False

        if eligible:
            results.append(scheme)
    return results
