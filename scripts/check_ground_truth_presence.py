"""
Check if ground truth assessments exist in metadata for COO query.
"""
import csv
import json
from pathlib import Path
from pprint import pprint
import pandas as pd

DATA_TRAIN = Path("data/train.csv")
ASSESSMENTS_CSV = Path("outputs/assessments_processed.csv")
ASSESSMENT_METADATA = Path("data/assessments_metadata.json")

def load_ground_truth_for_role(role_name_substr="COO"):
    rows = []
    with open(DATA_TRAIN, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            query = r.get("Query", "")
            if role_name_substr.lower() in query.lower():
                rows.append(r)
    return rows

def normalize_url(url):
    """Extract assessment slug from URL"""
    if '/view/' in url:
        slug = url.split('/view/')[-1].rstrip('/')
        return slug.lower()
    return url.lower().strip().rstrip('/')

def find_assessment_by_url(url, df):
    """Find assessment in dataframe by URL slug"""
    slug = normalize_url(url)
    
    # Try exact slug match
    for _, row in df.iterrows():
        assessment_url = str(row.get("url", ""))
        assessment_slug = normalize_url(assessment_url)
        
        if slug == assessment_slug or slug in assessment_slug or assessment_slug in slug:
            return row.to_dict()
    
    # Try partial name match (fallback)
    slug_words = slug.replace('-', ' ').split()
    for _, row in df.iterrows():
        name = str(row.get("name", "")).lower()
        if all(word in name for word in slug_words[:3]):  # Match first 3 words
            return row.to_dict()
    
    return None

def main():
    print("="*80)
    print("GROUND TRUTH PRESENCE CHECK FOR COO QUERY")
    print("="*80)
    
    # Load ground truth
    gt = load_ground_truth_for_role("COO")
    print(f"\nFound {len(gt)} ground-truth rows matching 'COO'")
    
    if not gt:
        print("❌ No ground-truth row for COO found in data/train.csv")
        return
    
    # Load assessments database
    if not ASSESSMENTS_CSV.exists():
        print(f"❌ Assessment CSV not found: {ASSESSMENTS_CSV}")
        return
    
    df = pd.read_csv(ASSESSMENTS_CSV)
    print(f"✅ Loaded {len(df)} assessments from {ASSESSMENTS_CSV}")
    
    # Get unique URLs from ground truth
    all_urls = set()
    for r in gt:
        url = r.get("Assessment_url", "").strip()
        if url:
            all_urls.add(url)
    
    print(f"\n{'='*80}")
    print(f"CHECKING {len(all_urls)} UNIQUE GROUND TRUTH URLS")
    print("="*80)
    
    found_count = 0
    missing_count = 0
    
    for url in sorted(all_urls):
        slug = normalize_url(url)
        assessment = find_assessment_by_url(url, df)
        
        if assessment:
            found_count += 1
            print(f"\n✅ FOUND: {slug}")
            print(f"   Name: {assessment.get('name', 'N/A')}")
            print(f"   URL: {assessment.get('url', 'N/A')}")
            print(f"   Type: {assessment.get('test_type', 'N/A')}")
            print(f"   Duration: {assessment.get('duration_minutes', 'N/A')} min")
        else:
            missing_count += 1
            print(f"\n❌ MISSING: {slug}")
            print(f"   Original URL: {url}")
            print(f"   NOT FOUND in 506-assessment database!")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Total ground truth URLs: {len(all_urls)}")
    print(f"Found in metadata: {found_count} ({found_count/len(all_urls)*100:.1f}%)")
    print(f"Missing from metadata: {missing_count} ({missing_count/len(all_urls)*100:.1f}%)")
    
    if missing_count > 0:
        print(f"\n⚠️  {missing_count} assessments are missing from the database!")
        print(f"   This explains poor recall - ground truth contains assessments we don't have.")
    else:
        print(f"\n✅ All ground truth assessments exist in metadata!")
        print(f"   The issue is retrieval/ranking, not missing data.")
    
    # Show sample query
    print(f"\n{'='*80}")
    print("SAMPLE GROUND TRUTH ROW")
    print("="*80)
    sample = gt[0]
    print(f"Query: {sample.get('Query', 'N/A')[:200]}...")
    print(f"Expected URL: {sample.get('Assessment_url', 'N/A')}")

if __name__ == "__main__":
    main()
