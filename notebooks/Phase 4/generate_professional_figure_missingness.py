import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def generate_professional_figure():
    # Path to feature classification file
    csv_path = r"data/processed/eda_results_corrected/feature_classification.csv"
    output_dir = r"notebooks/Phase 4/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate % Missing
    df['percent_missing'] = (1 - df['prevalence']) * 100
    
    # Clean feature names: remove suffix and format nicely
    df['Clean_Feature_Name'] = df['feature_name'].str.replace('_mean', '', regex=False).str.replace('_', ' ').str.title()
    
    # Calculate counts per category for the label
    category_counts = df['category'].value_counts()
    
    # Update category names to include (n=...)
    df['Category_Label'] = df['category'].apply(lambda x: f"{x}\n(n={category_counts[x]})")
    
    # Setup the professional style
    sns.set_style("whitegrid")
    sns.set_context("talk") # Larger fonts for readability
    
    # --- Figure 1: Distribution Boxplot ---
    plt.figure(figsize=(14, 10))
    
    # Define order by median missingness
    median_missing = df.groupby('category')['percent_missing'].median().sort_values()
    category_order = [f"{idx}\n(n={category_counts[idx]})" for idx in median_missing.index]
    
    # Use a more sophisticated, nature-friendly palette
    palette = sns.color_palette("RdYlBu", n_colors=len(category_order))
    
    # Create Horizontal Boxplot
    ax = sns.boxplot(
        x='percent_missing', 
        y='Category_Label', 
        data=df, 
        order=category_order, 
        palette=palette,
        linewidth=1.5,
        fliersize=0 # Hide outliers in boxplot to avoid duplication with stripplot
    )
    
    # Overlay Strip Plot (Jitter) to show individual data points
    sns.stripplot(
        x='percent_missing', 
        y='Category_Label', 
        data=df, 
        order=category_order, 
        size=4, 
        color=".3", 
        alpha=0.6,
        jitter=0.25
    )
    
    # Refine Axes
    plt.title("Feature Missingness by Category", fontsize=20, pad=20, weight='bold')
    plt.xlabel("Patients with Missing Data (%)", fontsize=16, weight='bold')
    plt.ylabel("") # Remove Y-axis label as categories are self-explanatory
    
    # Customize Ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-5, 105) # Ensure 0 and 100 are clearly visible
    
    # Despine for a cleaner look
    sns.despine(left=True, bottom=False)
    
    # Add vertical lines at key thresholds if useful (e.g. 50%)
    plt.axvline(x=50, color='gray', linestyle='--', alpha=0.3)
    
    # Save high-res PNG and PDF (vector)
    plt.tight_layout()
    png_path = os.path.join(output_dir, 'missingness_boxplot_professional.png')
    pdf_path = os.path.join(output_dir, 'missingness_boxplot_professional.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Figures saved to:\n{png_path}\n{pdf_path}")
    
    # --- Figure 2: Ranked Feature Completeness ---
    print("Generating ranked figure...")
    # Increase height significantly to accommodate 100+ features clearly
    plt.figure(figsize=(12, 24)) 
    
    # Sort by missingness
    df_sorted = df.sort_values('percent_missing', ascending=True)
    
    # Create barplot
    barplot = sns.barplot(
        x='percent_missing',
        y='Clean_Feature_Name',
        data=df_sorted,
        hue='category', # Color by category
        dodge=False,
        palette="RdYlBu" # Consistent palette
    )
    
    plt.title("Feature Missingness by Variable (Ranked)", fontsize=20, pad=20, weight='bold')
    plt.xlabel("Patients with Missing Data (%)", fontsize=16, weight='bold')
    plt.ylabel("")
    
    # Move legend to a better spot
    # Put legend inside plot, higher up
    plt.legend(title="Variable Category", bbox_to_anchor=(0.7, 0.5), loc='center left', frameon=True, fontsize=12)
    
    # Add gridlines for easier reading
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Ensure y-labels (feature names) are legible
    plt.yticks(fontsize=10) # Smaller font for many labels
    plt.xticks(fontsize=14)
    
    plt.tight_layout()
    ranked_path = os.path.join(output_dir, 'feature_missingness_ranked.pdf')
    # Save as PDF for vector quality (zooming in works)
    plt.savefig(ranked_path, bbox_inches='tight')
    
    # Also save a very high res PNG for quick preview
    ranked_png_path = os.path.join(output_dir, 'feature_missingness_ranked.png')
    plt.savefig(ranked_png_path, dpi=300, bbox_inches='tight')
    
    print(f"Alternative figure saved to {ranked_path} and {ranked_png_path}")

if __name__ == "__main__":
    generate_professional_figure()
