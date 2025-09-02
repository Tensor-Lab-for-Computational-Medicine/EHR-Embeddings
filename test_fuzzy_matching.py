#!/usr/bin/env python3

import pandas as pd
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

def build_loinc_search_index(vocab_dir: str = "./data/OMOP_Vocabulary") -> List[Dict]:
    """Build a comprehensive searchable index of LOINC concepts and synonyms."""
    print("Building LOINC search index from OMOP vocabulary...")
    
    # Load OMOP data
    concepts = pd.read_csv(f'{vocab_dir}/CONCEPT.csv', sep='\t', low_memory=False)
    synonyms = pd.read_csv(f'{vocab_dir}/CONCEPT_SYNONYM.csv', sep='\t', low_memory=False)
    
    # Get all standard LOINC concepts
    loinc_concepts = concepts[
        (concepts['vocabulary_id'] == 'LOINC') & 
        (concepts['standard_concept'] == 'S')
    ][['concept_id', 'concept_code', 'concept_name']]
    
    print(f"Found {len(loinc_concepts)} standard LOINC concepts")
    
    # Create searchable terms list
    searchable_terms = []
    
    # Add concept names
    for _, row in loinc_concepts.iterrows():
        if pd.notna(row['concept_name']):
            searchable_terms.append({
                'concept_id': row['concept_id'],
                'concept_code': row['concept_code'], 
                'searchable_name': row['concept_name'].lower(),
                'original_name': row['concept_name'],
                'source': 'concept_name'
            })
    
    # Add synonyms
    loinc_synonyms = synonyms[synonyms['concept_id'].isin(loinc_concepts['concept_id'])]
    synonym_count = 0
    
    for _, row in loinc_synonyms.iterrows():
        if pd.notna(row['concept_synonym_name']):
            concept_info = loinc_concepts[loinc_concepts['concept_id'] == row['concept_id']].iloc[0]
            searchable_terms.append({
                'concept_id': row['concept_id'],
                'concept_code': concept_info['concept_code'],
                'searchable_name': row['concept_synonym_name'].lower(),
                'original_name': row['concept_synonym_name'],
                'source': 'synonym'
            })
            synonym_count += 1
    
    print(f"Added {synonym_count} LOINC synonyms")
    print(f"Total searchable terms: {len(searchable_terms)}")
    
    return searchable_terms

def find_best_loinc_match(mimic_name: str, searchable_terms: List[Dict], min_score: float = 0.4) -> Tuple[Optional[Dict], float]:
    """Find best matching LOINC concept for a MIMIC lab name using fuzzy matching."""
    
    # Clean up MIMIC name for matching
    cleaned_mimic = mimic_name.replace('_mean', '').replace('_', ' ').lower()
    
    best_score = 0
    best_match = None
    
    for term in searchable_terms:
        # Calculate similarity score using SequenceMatcher
        score = SequenceMatcher(None, cleaned_mimic, term['searchable_name']).ratio()
        
        # Boost score for exact word matches
        mimic_words = set(cleaned_mimic.split())
        term_words = set(term['searchable_name'].split())
        
        if mimic_words.intersection(term_words):
            word_overlap = len(mimic_words.intersection(term_words)) / len(mimic_words.union(term_words))
            score = max(score, word_overlap * 0.8)  # Word overlap bonus
        
        if score > best_score and score >= min_score:
            best_score = score
            best_match = term
    
    return best_match, best_score

def test_fuzzy_matching():
    """Test the fuzzy matching approach on sample MIMIC lab names."""
    print("Testing fuzzy matching approach for MIMIC lab mappings...\n")
    
    # Build search index
    searchable_terms = build_loinc_search_index()
    
    # Test with sample MIMIC lab names
    test_labs = [
        'albumin_mean', 
        'glucose_mean', 
        'creatinine_mean', 
        'potassium_mean', 
        'hemoglobin_mean',
        'alanine aminotransferase_mean',
        'alkaline phosphate_mean',
        'blood urea nitrogen_mean',
        'white blood cells_mean',
        'cardiac output fick_mean'  # More complex case
    ]
    
    print("Fuzzy matching results:")
    print("=" * 80)
    
    successful_matches = 0
    
    for lab in test_labs:
        match, score = find_best_loinc_match(lab, searchable_terms)
        if match:
            print(f"{lab:30} -> LOINC/{match['concept_code']} (score: {score:.3f})")
            print(f"{'':32}   {match['original_name'][:60]}...")
            successful_matches += 1
        else:
            print(f"{lab:30} -> No match found")
        print()
    
    print(f"Successfully matched {successful_matches}/{len(test_labs)} lab types")
    print(f"Match rate: {successful_matches/len(test_labs)*100:.1f}%")

if __name__ == "__main__":
    test_fuzzy_matching() 