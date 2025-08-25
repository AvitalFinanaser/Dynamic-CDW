import os
import random
import pandas as pd
from typing import Dict
import json
from collections import defaultdict

# Define our configuration constants with continuous ranges for all possible values
DOMAIN_MAPPINGS = {
    "building": {"prefix": "BLD", "name": "Building"},
    "transport": {"prefix": "TRN", "name": "Transport"},
    "energy": {"prefix": "ENG", "name": "Energy"},
    "waste": {"prefix": "WST", "name": "Waste"},
    "water": {"prefix": "WTR", "name": "Water"},
    "health": {"prefix": "HTH", "name": "Health"},
    "business": {"prefix": "BUS", "name": "Business"},
    "natural_environment": {"prefix": "ENV", "name": "Natural Environment"},
    "land_use": {"prefix": "LND", "name": "Land Use"},
    "natural_hazards": {"prefix": "HZD", "name": "Natural Hazards"}
}

# Define sentiment categories within range [0,1]
SENTIMENT_CATEGORIES = {
    (0.00, 0.20): "active_resistance",  # Complete opposition to climate action
    (0.20, 0.40): "minimal_acknowledgment",  # Basic recognition without meaningful action
    (0.40, 0.60): "balanced_approach",  # Equal consideration of climate and other factors
    (0.60, 0.80): "supportive_measures",  # Proactive support with practical constraints
    (0.80, 1.00): "proactive_action"  # Climate-first transformative approaches
}


def get_sentiment_category(sentiment: float) -> str:
    """
    Determine the sentiment category for any value between 0 and 1.
    This function ensures every possible sentiment value has a category.
    """
    if sentiment < 0.00 or sentiment > 1.00:
        return "undefined"

    # Handle the exact boundary cases
    if sentiment == 1.00:
        return "proactive_action"

    # Find the appropriate range for the sentiment value
    for (lower, upper), category in SENTIMENT_CATEGORIES.items():
        if lower <= sentiment < upper:
            return category

    return "undefined"


def create_proposal_id(domain_prefix: str, sentiment: float) -> str:
    """
    Create a unique identifier for each proposal.
    Example: BLD_000 for Building domain with 0.00 sentiment
    """
    sentiment_str = f"{int(sentiment * 100):03d}"
    return f"{domain_prefix}_{sentiment_str}"


def create_metadata() -> Dict:
    """
    Create the metadata section describing the dataset structure and categories.
    This helps users understand how to interpret the sentiment values.
    """
    return {
        "description": "Comprehensive Municipal Climate Action Proposals Dataset",
        "purpose": "Synthetic dataset - collaborative constitution writing for municipal climate action",
        "domains": len(DOMAIN_MAPPINGS),
        "sentiment_scale": {
            "range": "0.00 to 1.00",
            "increment": 0.05,
            "levels": 21,
            "categories": {
                "active_resistance": {
                    "range": "0.00-0.20",
                    "description": "Positions that actively resist or oppose climate action"
                },
                "minimal_acknowledgment": {
                    "range": "0.20-0.40",
                    "description": "Positions that acknowledge but minimize climate considerations"
                },
                "balanced_approach": {
                    "range": "0.40-0.60",
                    "description": "Positions balancing climate with other factors"
                },
                "supportive_measures": {
                    "range": "0.60-0.80",
                    "description": "Positions actively supporting climate action with practical constraints"
                },
                "proactive_action": {
                    "range": "0.80-1.00",
                    "description": "Positions making climate action a primary priority"
                }
            }
        }
    }


def process_domain_file(domain_id: str, file_content: str) -> Dict:
    """
    Process a single domain's JSON content and standardize its format.
    Handles the conversion of proposals while maintaining data integrity.
    """
    try:
        # Parse the domain-specific JSON
        domain_data = json.loads(file_content)

        # Extract domain information
        domain_info = {
            "id": domain_id,
            "name": DOMAIN_MAPPINGS[domain_id]["name"],
            "description": domain_data.get("description", "")
        }

        # Process each proposal in the domain
        items = []
        proposals = domain_data.get("proposals", [])

        for proposal in proposals:
            sentiment = float(proposal["sentiment"])
            item = {
                "id": create_proposal_id(DOMAIN_MAPPINGS[domain_id]["prefix"], sentiment),
                "text": proposal["text"],
                "sentiment": sentiment,
                "sentiment_category": get_sentiment_category(sentiment),
                "rationale": proposal["rationale"]
            }
            items.append(item)

        return {
            "domain": domain_info,
            "items": sorted(items, key=lambda x: x["sentiment"])  # Sort by sentiment score
        }

    except Exception as e:
        print(f"Error processing domain {domain_id}: {str(e)}")
        return None


def combine_climate_proposals(input_files: Dict[str, str]) -> Dict:
    """
    Combine multiple domain JSON files into a single unified structure.
    Creates a consistent format across all domains.
    """
    unified_data = {
        "metadata": create_metadata(),
        "proposals": []
    }

    # Process each domain and add to the unified structure
    for domain_id, file_content in sorted(input_files.items()):  # Sort domains alphabetically
        processed_domain = process_domain_file(domain_id, file_content)
        if processed_domain:
            unified_data["proposals"].append(processed_domain)

    return unified_data


def combine():
    """
    Main function to read files and create the unified dataset.
    Handles file reading, processing, and output generation.
    """
    # List of all domain files to process
    file_list = [
        ("building", "municipal-climate-dataset-building.json"),
        ("transport", "municipal-climate-dataset-transport.json"),
        ("energy", "municipal-climate-dataset-energy.json"),
        ("waste", "municipal-climate-dataset-waste.json"),
        ("water", "municipal-climate-dataset-water.json"),
        ("health", "municipal-climate-dataset-health.json"),
        ("business", "municipal-climate-dataset-business.json"),
        ("natural_environment", "municipal-climate-dataset-natural_environment.json"),
        ("land_use", "municipal-climate-dataset-land_use.json"),
        ("natural_hazards", "municipal-climate-dataset-natural_hazards.json")
    ]

    # Read all domain files
    domain_files = {}
    for domain_id, filename in file_list:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                domain_files[domain_id] = f.read()
            print(f"Successfully read {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            continue

    # Combine all domains into unified dataset
    unified_dataset = combine_climate_proposals(domain_files)

    # Save the result
    output_file = 'unified_synthetic_climate_proposals.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unified_dataset, f, indent=2, ensure_ascii=False)
    print(f"Successfully created {output_file}")


def df_JSON(json_location: str):
    """
    Creates a data frame from the JSON file
    """
    # Load the JSON data
    with open(json_location, 'r') as file:
        data = json.load(file)

    # Extracting the relevant information
    proposals = []
    for domain in data.get("proposals", []):
        domain_id = domain["domain"]["id"]
        domain_name = domain["domain"]["name"]
        domain_description = domain["domain"].get("description", "")
        for item in domain.get("items", []):
            proposals.append({
                "Domain ID": domain_id,
                "Domain Name": domain_name,
                "Domain Description": domain_description,
                "Proposal ID": item["id"],
                "Proposal Text": item["text"],
                "Sentiment": item["sentiment"],
                "Sentiment Category": item["sentiment_category"],
                "Rationale": item["rationale"]
            })

    # Create a DataFrame
    df_proposals = pd.DataFrame(proposals)
    return df_proposals


def synthetic_climate_proposals_sentiment():
    """
    Loads the unified synthetic climate proposals dataset and groups proposals by sentiment category.

    Returns:
    --------
    grouped_proposals : defaultdict(list)
        A dictionary where each key is a sentiment category (e.g., 'balanced_approach', 'proactive_action'),
        and the value is a list of proposal dictionaries with:
            - text: str, the proposal text
            - domain: str, the domain category (e.g., 'Energy', 'Health')
            - rationale: str, the reasoning behind the proposal
    """
    # Load the JSON dataset
    file_path = os.path.join('datasets', 'synthetic climate', 'unified_synthetic_climate_proposals.json')

    with open(file_path, 'r') as f:
        data = json.load(f)

    proposals_data = data["proposals"]

    # Group by sentiment_category
    grouped_proposals = defaultdict(list)

    for domain_entry in proposals_data:
        domain_name = domain_entry['domain']['name']
        for item in domain_entry['items']:
            category = item['sentiment_category']
            grouped_proposals[category].append({
                "text": item['text'],
                "domain": domain_name,
                "rationale": item['rationale']
            })

    return grouped_proposals


def sample_diverse_proposals(grouped_proposals, category, num_samples=3):
    """
    Samples a specified number of proposals from a sentiment category ensuring different domains.

    Parameters:
    -----------
    grouped_proposals : dict
        Dictionary of proposals grouped by sentiment_category as returned from synthetic_climate_proposals_sentiment.

    category : str
        The sentiment category to sample from (e.g., 'balanced_approach').

    num_samples : int, optional (default=3)
        The number of diverse domain proposals to return.

    Returns:
    --------
    diverse_samples : list of dicts
        List containing sampled proposal dictionaries with keys:
            - text: str, the proposal text
            - domain: str, the domain category
            - rationale: str, the reasoning behind the proposal
    """
    proposals = grouped_proposals[category]
    random.shuffle(proposals)

    seen_domains = set()
    diverse_samples = []

    for proposal in proposals:
        if proposal['domain'] not in seen_domains:
            diverse_samples.append(proposal)
            seen_domains.add(proposal['domain'])
        if len(diverse_samples) == num_samples:
            break

    return diverse_samples


if __name__ == "__main__":
    # Combining climate domain files
    # combine()
    # # Creating a Data Frame from JSON file
    # df = df_JSON('unified_synthetic_climate_proposals.json')
    # df.to_csv('unified_synthetic_climate_proposals.csv', index=False)
    # Group by sentiment
    file_path = os.path.join('datasets', 'synthetic climate', 'unified_synthetic_climate_proposals.json')
    grouped_proposals = synthetic_climate_proposals_sentiment()
    sampled = sample_diverse_proposals(grouped_proposals, 'balanced_approach', num_samples=3)
    print(sampled)
    # Saving grouped dataset JSON file
    save_path = os.path.join('datasets', 'processed', 'grouped_proposals_by_sentiment.json')
    with open(save_path, 'w') as f:
        json.dump(grouped_proposals, f, indent=4)



