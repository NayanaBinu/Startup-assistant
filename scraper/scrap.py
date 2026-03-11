import json
import re
import logging

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ---------------- Utility Functions ---------------- #
def clean_html(raw_html: str) -> str:
    """Removes HTML tags from a string safely."""
    if not raw_html:
        return ""
    cleanr = re.compile(r'<[^>]+>')
    return re.sub(cleanr, '', raw_html).strip()


def map_eligibility(full_text: str, eligibility_text: str):
    """
    Map eligibility text into structured categories:
    domain, registration, and stage.
    """
    eligibility_data = {"domain": [], "registration": [], "stage": []}

    # 1. Domain Mapping
    domain_map = {
        "agriculture": ["agri", "agtech", "farm", "animal husbandry", "aquaculture"],
        "biotech": ["bio", "health", "medical", "pharma"],
        "tech": ["tech", "it", "electronics", "digital", "software", "hardware"],
        "manufacturing": ["manufacturing", "industrial", "production"],
        "services": ["service", "trading", "tourism"],
        "education": ["education", "skill"],
        "social impact": ["social", "women", "sc/st", "tribal", "backward class"]
    }
    for domain, keywords in domain_map.items():
        if any(keyword.lower() in full_text for keyword in keywords):
            eligibility_data["domain"].append(domain)
    if not eligibility_data["domain"]:
        eligibility_data["domain"].append("all")

    # 2. Registration Mapping
    reg_map = {
        "private limited": ["private limited", "company", "companies"],
        "LLP": ["llp", "limited liability"],
        "MSME": ["msme", "micro", "small", "medium enterprise"],
        "society": ["society", "cooperative"],
        "trust": ["trust"]
    }
    for reg_type, keywords in reg_map.items():
        if any(keyword.lower() in eligibility_text for keyword in keywords):
            eligibility_data["registration"].append(reg_type)
    if not eligibility_data["registration"]:
        eligibility_data["registration"] = list(reg_map.keys())

    # 3. Stage Mapping
    stage_map = {
        "early": ["early", "new", "start-up", "startups", "innovators", "seed"],
        "growth": ["growth", "expand", "expansion", "existing"],
        "scaling": ["scaling", "scale"]
    }
    for stage, keywords in stage_map.items():
        if any(keyword.lower() in full_text for keyword in keywords):
            eligibility_data["stage"].append(stage)
    if not eligibility_data["stage"]:
        eligibility_data["stage"] = list(stage_map.keys())

    return {
        "domain": list(set(eligibility_data["domain"])),
        "registration": list(set(eligibility_data["registration"])),
        "stage": list(set(eligibility_data["stage"]))
    }


# ---------------- Main Processing ---------------- #
def process_raw_data(input_file="source_data.json", output_file="startup_schemes_final.json"):
    """Reads nested source JSON, flattens it, and assigns eligibility criteria."""

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading source file {input_file}: {e}")
        return

    schemes_data = data.get("data", {}).get("searchResult", {})
    output_list = []

    logging.info(f"Found {len(schemes_data)} ministries/organizations. Processing schemes...")

    processed_scheme_ids = set()  # to avoid duplicates

    for ministry, schemes in schemes_data.items():
        if not schemes:
            continue

        for scheme in schemes:
            # Extract scheme name
            name_list = scheme.get("schname")
            if not name_list or not name_list[0]:
                continue
            name = name_list[0].strip()

            # Unique ID by ministry + scheme name
            scheme_id = (ministry, name)
            if scheme_id in processed_scheme_ids:
                continue
            processed_scheme_ids.add(scheme_id)

            # Benefits
            benefits_list = scheme.get("benefits") or []
            benefits = [clean_html(b).strip() for b in benefits_list if b]
            if not benefits:
                benefits = ["Details not provided."]

            # Link
            link = (scheme.get("linktoApplication") or [None])[0] or "#"

            # Eligibility texts
            eligibility_criteria_list = scheme.get("EligibilityCriteria") or []
            eligibility_text = " ".join(
                [clean_html(e).strip() for e in eligibility_criteria_list if e]
            ).lower()
            sector_text = " ".join(scheme.get("sector") or []).lower()
            brief_text = " ".join(scheme.get("brief") or []).lower()

            full_text_for_search = f"{eligibility_text} {sector_text} {brief_text}"

            # Map eligibility
            eligibility_data = map_eligibility(full_text_for_search, eligibility_text)

            # Build final scheme object
            new_scheme = {
                "name": name,
                "benefits": benefits,
                "link": link,
                "eligibility": eligibility_data,
                "raw_eligibility": [clean_html(e).strip() for e in eligibility_criteria_list if e]
            }
            output_list.append(new_scheme)

    # Save final JSON
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_list, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully processed and saved {len(output_list)} unique schemes to {output_file}.")
    except Exception as e:
        logging.error(f"Error writing output file {output_file}: {e}")


# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    process_raw_data("source_data.json", "startup_schemes_final.json")
