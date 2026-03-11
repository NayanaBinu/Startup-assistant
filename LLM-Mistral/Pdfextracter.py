import fitz
import re

doc = fitz.open("A2013-18.pdf")
full_text = ""

# Combine entire PDF text
for page in doc:
    full_text += page.get_text()

# Use regex to extract section blocks
pattern = r"\n?\s*(\d+[A-Z]?)\.?\s+(.*?)(?=\n\s*\d+[A-Z]?\.\s|\Z)"  # Match section + title + text
matches = re.findall(pattern, full_text, re.DOTALL)

sections = []

for i, (sec_num, content) in enumerate(matches):
    lines = content.strip().split("\n", 1)
    title = lines[0].strip()
    body = lines[1].strip() if len(lines) > 1 else ""
    
    sections.append({
        "section": f"Section {sec_num}",
        "title": title,
        "text": body
    })

# Save as JSON
import json
with open("companies_act_sections.json", "w") as f:
    json.dump(sections, f, indent=2)
