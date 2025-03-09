# Execute this file from top-level.
# Do not execute it from local directory.

import json

# File paths
input_file = "data/raw/eng.txt"
output_file = "data/processed/eng_v.json"

# Dictionary to store verb paradigms
verb_dict = {}

# Process the file line by line
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        base, inflected, form = line.strip().split("\t")

        # Skip if it's a noun
        if not form.startswith("V;"):
            continue

        # Initialize dictionary entry if not exists
        if base not in verb_dict:
            verb_dict[base] = {
                "PRS": base,  # Base form is the present verb
                "3SG": None,
                "PST": None,
                "PRS.PTCP": None,
                "PST.PTCP": None,
            }

        # Assign inflected forms to the appropriate slot
        if form == "V;PRS;3;SG":
            verb_dict[base]["3SG"] = inflected
        elif form == "V;PST":
            verb_dict[base]["PST"] = inflected
        elif form == "V;V.PTCP;PRS":
            verb_dict[base]["PRS.PTCP"] = inflected
        elif form == "V;V.PTCP;PST":
            verb_dict[base]["PST.PTCP"] = inflected

# Remove incomplete paradigms
full_paradigms = {
    base: forms for base, forms in verb_dict.items() 
    if all(forms.values())  # Ensure no `None` values
}

# Write to JSON
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(full_paradigms, f_out, ensure_ascii=False, indent=2)

print(f"Processed verb paradigms saved to {output_file}")
