import os
import csv
import logging
from pathlib import Path
import pandas as pd
import sys

# Add parent directory to path for utility imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from manuscript_table_utils import save_manuscript_html
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RULES_CSV = BASE_DIR / 'notebooks' / 'Phase_6' / 'feature_engineering' / 'feature_rules.csv'
OUTPUT_DIR = BASE_DIR / 'manuscript_figures'

def setup_environment() -> None:
    """Set up logging configuration and plot styling."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    # Manuscript-friendly defaults
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.titlesize': 14,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

def clean_feature_name(feat_name: str) -> str:
    """Clean the raw EHR feature names for nice visualization."""
    if '[All' in feat_name:
        return 'All Data Density'
    
    clean = feat_name.replace('_mean_last', '').replace('_mean_count', '')
    clean = clean.replace('_mean_slope_24h', '').replace('_mean_stddev_24h', '')
    clean = clean.replace('_encoded', '')
    clean = clean.replace('_', ' ').title()
    
    # Specific clinical terminology mappings
    fixes = {
        'Glascow Coma Scale Total': 'Glasgow Coma Scale',
        'Ph': 'pH',
        'Sirs': 'SIRS',
        'Troponin-I': 'Troponin I',
        'Troponin-T': 'Troponin T',
        'Co2': 'CO2',
        'Fio2': 'FiO2',
        'Pao2': 'PaO2',
        'Paco2': 'PaCO2',
    }
    for k, v in fixes.items():
        if clean == k or k in clean:
            clean = clean.replace(k, v)
    return clean

def load_phenotype_rules(csv_path: Path) -> pd.DataFrame:
    """Parse the feature rules CSV, extracting category information from comments."""
    rows = []
    current_category = "General"
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
            
        for row in reader:
            # Skip empty rows
            if not row or not row[0].strip():
                continue
            
            # Extract category from comment headers
            if row[0].startswith('#'):
                cat = row[0].replace('#', '').replace('---', '').strip()
                if 'PHENOTYPES' in cat:
                    cat = cat.replace('PHENOTYPES', '').strip()
                elif 'CRITERIA COMPONENTS' in cat:
                    cat = cat.replace('CRITERIA COMPONENTS', '').strip()
                
                # Title case but fix small words
                current_category = cat.title().replace(' And ', ' & ')
                continue
            
            if len(row) >= 5:
                phenotype_name = row[0].strip()
                if phenotype_name == 'phenotype_name': # Header skip safeguard
                    continue
                    
                record = {
                    'phenotype_name': phenotype_name,
                    'phenotype_type': row[1].strip().title(),
                    'logic': row[2].strip(),
                    'rationale': row[3].strip(),
                    'required_features': row[4].strip(),
                    'category': current_category
                }
                rows.append(record)
                
    return pd.DataFrame(rows)

def plot_phenotype_overview(df: pd.DataFrame) -> None:
    """Generate a two-panel summary figure of the phenotypes."""
    logging.info("Generating phenotype overview figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Subplot 1: Stacked Bar Chart by Category and Type ---
    # Prepare data
    cat_type_counts = df.groupby(['category', 'phenotype_type']).size().unstack(fill_value=0)
    cat_type_counts['Total'] = cat_type_counts.sum(axis=1)
    cat_type_counts = cat_type_counts.sort_values('Total', ascending=True).drop('Total', axis=1)
    
    # Plot
    colors = sns.color_palette("Set2", n_colors=len(cat_type_counts.columns))
    cat_type_counts.plot(kind='barh', stacked=True, ax=axes[0], color=colors, edgecolor='black', linewidth=0.5)
    
    axes[0].set_title('A. Phenotypes by Clinical System & Data Type', loc='left', fontweight='bold')
    axes[0].set_xlabel('Number of Engineered Phenotypes')
    axes[0].set_ylabel('Clinical System / Category')
    axes[0].legend(title='Data Type', loc='lower right', frameon=True)
    
    # --- Subplot 2: Top Features Utilized ---
    all_features = []
    for feats in df['required_features'].dropna():
        for f in feats.split('|'):
            clean_f = clean_feature_name(f.strip())
            # Don't count other phenotypes as base features for this plot if they are just self-referential
            if clean_f != '' and clean_f.lower().replace(' ', '_') not in df['phenotype_name'].values:
                all_features.append(clean_f)
            
    feat_series = pd.Series(all_features)
    # Count unique usages (how many phenotypes use each feature)
    top_features = feat_series.value_counts().head(15).sort_values(ascending=True)
    
    # Plot
    bar_colors = sns.color_palette("mako", n_colors=15)
    axes[1].barh(top_features.index, top_features.values, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    axes[1].set_title('B. Top Source EHR Features Driving Phenotypes', loc='left', fontweight='bold')
    axes[1].set_xlabel('Number of Phenotypes Utilizing Feature')
    axes[1].set_ylabel('Source EHR Feature')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'figure_phenotype_overview.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved overview to {save_path}")

def plot_phenotype_heatmap(df: pd.DataFrame) -> None:
    """Generate a heatmap mapping raw EHR features to the engineered phenotypes."""
    logging.info("Generating phenotype-feature heatmap...")
    
    # Build mapping
    mapping = []
    pheno_names_raw = df['phenotype_name'].values
    
    for _, row in df.iterrows():
        phenotype = row['phenotype_name'].replace('_', ' ').title()
        features = row['required_features'].split('|')
        for f in features:
            f_clean = clean_feature_name(f.strip())
            
            # Avoid showing phenotypes that simply rely on other phenotypes directly in this raw feature map 
            # (unless desired, but keeping it to base EHR features is cleaner)
            f_raw = f.strip()
            if f_raw in pheno_names_raw:
                continue
                
            mapping.append({
                'Phenotype': phenotype,
                'Feature': f_clean,
                'Category': row['category']
            })
            
    map_df = pd.DataFrame(mapping)
    if map_df.empty:
        logging.warning("No features mapped for heatmap.")
        return
        
    # Create Matrix
    matrix = pd.crosstab(map_df['Feature'], map_df['Phenotype'])
    
    # Sort features (rows) by total occurrence
    matrix['total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values('total', ascending=False).drop('total', axis=1)
    
    # Group phenotypes (columns) by Category
    cat_order = df['category'].unique()
    col_order = []
    col_categories = []
    
    for cat in cat_order:
        # Get phenotypes in this category, formatted as title case
        phenos = df[df['category'] == cat]['phenotype_name'].apply(lambda x: x.replace('_', ' ').title()).tolist()
        # Only keep ones that are in our matrix
        valid_phenos = [p for p in phenos if p in matrix.columns]
        col_order.extend(valid_phenos)
        col_categories.extend([cat] * len(valid_phenos))
        
    matrix = matrix[col_order]
    
    # --- Plotting ---
    fig = plt.figure(figsize=(22, 14))
    
    # Adjust gridspec to accommodate a color bar for categories at the top
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 20], hspace=0.01)
    ax_color = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    
    # 1. Plot Heatmap
    # Use a binary colormap: 0=white, 1=navy blue
    cmap = sns.color_palette(["#ffffff", "#1f4e79"])
    sns.heatmap(matrix > 0, cmap=cmap, cbar=False, linewidths=0.5, linecolor='#e0e0e0', ax=ax_heat)
    
    ax_heat.set_xlabel('')
    ax_heat.set_ylabel('Source EHR Features', fontsize=14, fontweight='bold', labelpad=15)
    
    # Rotate x-axis labels for readability
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), fontsize=11)
    
    # 2. Plot Category Color Bar
    unique_cats = list(dict.fromkeys(col_categories)) # Preserve order roughly
    cat_colors = sns.color_palette("husl", len(unique_cats))
    color_map = dict(zip(unique_cats, cat_colors))
    
    # Numeric array for colormap
    cat_numeric = [unique_cats.index(c) for c in col_categories]
    ax_color.imshow([cat_numeric], aspect='auto', cmap=ListedColormap(cat_colors), interpolation='none')
    ax_color.axis('off') # Hide axes for color bar
    
    # 3. Add Legend for Categories
    patches = [mpatches.Patch(color=color_map[cat], label=cat) for cat in unique_cats]
    ax_color.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 1.2), 
                    ncol=min(len(unique_cats), 5), frameon=False, fontsize=12)
    
    plt.suptitle('Mapping of Source EHR Features to Engineered Phenotypes', fontsize=18, fontweight='bold', y=0.98)
    
    # Add borders to the heatmap
    for _, spine in ax_heat.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
        
    save_path = OUTPUT_DIR / 'figure_phenotype_feature_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved heatmap to {save_path}")

def generate_dictionary_tables(df: pd.DataFrame) -> None:
    """Generate Markdown and LaTeX tables summarizing the phenotype dictionary."""
    logging.info("Generating phenotype dictionary tables...")
    
    # Prepare clean DataFrame
    export_df = df.copy()
    export_df['phenotype_name'] = export_df['phenotype_name'].apply(lambda x: x.replace('_', ' ').title())
    
    # Clean up required features
    def clean_feats(feat_str):
        feats = [clean_feature_name(f.strip()) for f in feat_str.split('|')]
        # Unique and non-empty
        return ", ".join(sorted(list(set([f for f in feats if f]))))
        
    export_df['required_features'] = export_df['required_features'].apply(clean_feats)
    export_df = export_df[['category', 'phenotype_name', 'phenotype_type', 'rationale', 'required_features']]
    export_df.columns = ['Clinical System', 'Phenotype', 'Data Type', 'Rationale', 'Key Source Features']
    
    export_df = export_df.sort_values('Clinical System')
    
    # --- Markdown ---
    md_path = OUTPUT_DIR / 'table_phenotype_dictionary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Engineered Phenotype Dictionary\n\n")
        f.write(export_df.to_markdown(index=False))
            
        
    # --- HTML (for Google Docs) ---
    save_manuscript_html(
        export_df,
        "Engineered Phenotype Dictionary Mapping",
        "table_phenotype_dictionary",
        OUTPUT_DIR
    )
    logging.info(f"Professional HTML table saved to {OUTPUT_DIR / 'table_phenotype_dictionary.html'}")
    
    logging.info(f"Saved dictionary tables to {OUTPUT_DIR}")

def main():
    setup_environment()
    
    if not RULES_CSV.exists():
        logging.error(f"Cannot find rules CSV at {RULES_CSV}")
        return
        
    df = load_phenotype_rules(RULES_CSV)
    logging.info(f"Loaded {len(df)} phenotypes across {df['category'].nunique()} clinical systems.")
    
    plot_phenotype_overview(df)
    plot_phenotype_heatmap(df)
    generate_dictionary_tables(df)
    logging.info("All phenotype visualizations generated successfully.")

if __name__ == '__main__':
    main()
