import csv
import re
import pandas as pd

def get_vars(f):
    df = pd.read_csv(f)
    rules = df['Archetype rule'].tolist()
    vars_set = set()
    for r in rules:
        parts = r.split(' AND ')
        for p in parts:
            match = re.match(r'([a-zA-Z0-9_]+)[\=\<\>]+', p)
            if match:
                vars_set.add(match.group(1))
            elif ':' in p:
                vars_set.add(p.split(':')[0].strip())
    return vars_set

m_vars = get_vars('notebooks/Phase 6 - H2 Analysis/h2b/outputs/mortality/tables/top_archetypes_mortality_final_archetypes_ivb.csv')
r_vars = get_vars('notebooks/Phase 6 - H2 Analysis/h2b/outputs/readmission_30/tables/top_archetypes_readmission_30_final_archetypes_ivb.csv')

all_rule_vars = m_vars.union(r_vars)

raw_vars = set()
with open('notebooks/Phase 6 - H2 Analysis/feature_engineering/feature_rules.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row or row[0].startswith('#'): continue
        if row[0] in all_rule_vars:
            req = row[-1]
            raw_vars.update(req.split('|'))

# Find the ones we are plotting
with open('notebooks/Phase 4/map_clinical_to_embeddings_v4.py') as f:
    script = f.read()

curated_match = re.search(r'CURATED_VARIABLES = \[(.*?)\]', script, re.DOTALL)
if curated_match:
    curated = [x.strip().strip("'").strip('"') for x in curated_match.group(1).split(',')]
    curated = [x for x in curated if x]
    
    print('Raw variables required by rules:', sorted(list(raw_vars)))
    print('\nMissing from CURATED_VARIABLES:')
    missing = [v for v in raw_vars if v not in curated and not v.endswith('_count')]
    print(missing)
