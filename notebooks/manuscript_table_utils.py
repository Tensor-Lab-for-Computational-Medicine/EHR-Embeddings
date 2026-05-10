import pandas as pd
from pathlib import Path
import re
import numpy as np
import math

import os

def format_3_sig_figs(val):
    """Formats a number to exactly 3 significant figures."""
    if val is None or pd.isna(val):
        return "—"
    try:
        num = float(val)
    except:
        return str(val)
    
    if num == 0:
        return "0.00"
    
    abs_num = abs(num)
    
    # Calculate magnitude
    import math
    mag = math.floor(math.log10(abs_num))
    
    # If magnitude is very small (e.g. < 0.001), use scientific notation
    if mag <= -3:
        fmt_val = f"{num:.2e}"
        if 'e' in fmt_val:
            m, e = fmt_val.split('e')
            return f"{m}&nbsp;&times;&nbsp;10<sup>{int(e)}</sup>"
        return fmt_val
    else:
        # Number of decimal places needed to get 3 sig figs
        decimals = max(0, 2 - mag)
        return f"{num:.{decimals}f}"

# Global Design Tokens for consistency
FONT_FAMILY = "'Times New Roman', serif"
TEXT_SIZE = "10pt"
HEADER_SIZE = "11pt"
TITLE_SIZE = "12pt"
BORDER_COLOR = "black"
ROW_ALT_COLOR = "#fbfbfb" # Very subtle zebra striping
SIGNIFICANCE_THRESHOLD = 0.05

# Centralized Output Paths - Can be overridden via environment variables
# Default: assumes project structure with manuscript_outputs at the same level as the parent of this file.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "manuscript_outputs"
ROOT_OUTPUT_DIR = Path(os.getenv("MANUSCRIPT_OUTPUT_DIR", DEFAULT_ROOT))

TABLES_OUTPUT_DIR = ROOT_OUTPUT_DIR / "Tables"
FIGURES_OUTPUT_DIR = ROOT_OUTPUT_DIR / "Figures"

# Ensure directories exist
TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_manuscript_html(df, title, filename, output_dir, footnotes=None, compact=False, table_number=None):
    """
    Generates a professional 'Three-Line' manuscript table in HTML format.
    
    Improvements:
    1.  Significance Highlighting: Bolds q-values < 0.05.
    2.  Zebra Striping: Subtle background for alternate rows.
    3.  Footnotes: Support for explanatory notes at the bottom.
    4.  Smart Alignment: Centers small integers, right-aligns floats, left-aligns text.
    5.  Text Wrapping: Capped width for long text columns.
    6.  Null Handling: Standardized '—' for missing values.
    7.  Compact Mode: Reduced padding and font size for large supplements.
    8.  Modular Styling: Centralized design tokens.
    9.  Scientific Notation: Refined math-style formatting.
    10. Title Numbering: Optional 'Table X' prefixing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust sizes for compact mode
    text_sz = "8.5pt" if compact else TEXT_SIZE
    padding = "1pt 4pt" if compact else "2.5pt 6pt"
    
    # CSS styles
    top_border = f"border-top: 1.5pt solid {BORDER_COLOR};"
    header_border = f"border-bottom: 1pt solid {BORDER_COLOR};"
    bottom_border = f"border-bottom: 1.5pt solid {BORDER_COLOR};"
    
    base_header = f"text-align: left; padding: 3pt 6pt; font-weight: bold; font-family: {FONT_FAMILY}; font-size: {HEADER_SIZE};"
    base_cell = f"border: none; padding: {padding}; font-family: {FONT_FAMILY}; font-size: {text_sz}; vertical-align: top; line-height: 1.15;"
    group_header_style = f"{base_cell} font-weight: bold; padding-top: 6pt; border-bottom: 0.5pt solid #eee;"

    # Determine grouping column
    group_col = next((c for c in ["Comparison", "Task", "Analysis"] if c in df.columns), None)
    display_cols = [c for c in df.columns if c != group_col]
    
    # Prepare Title
    full_title = f"Table {table_number}: {title}" if table_number else title
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <style>
            table {{ border-collapse: collapse; width: 100%; max-width: 1200px; font-family: {FONT_FAMILY}; }}
            .text-cell {{ text-align: left; max-width: 400px; word-wrap: break-word; }}
            .num-cell {{ text-align: right; white-space: nowrap; }}
            .center-cell {{ text-align: center; }}
            .sig-value {{ font-weight: bold; }}
            .footnote {{ font-size: 8pt; margin-top: 4pt; font-family: {FONT_FAMILY}; line-height: 1.1; color: #444; }}
            tr:nth-child(even) {{ background-color: {ROW_ALT_COLOR}; }}
            tr.group-row {{ background-color: white !important; }}
        </style>
    </head>
    <body style="padding: 20px;">
        <h3 style="font-size: {TITLE_SIZE}; margin-bottom: 4pt; font-weight: bold;">{full_title}</h3>
        <table style="{bottom_border}">
            <thead>
                <tr>
                    {" ".join(f'<th style="{base_header} {top_border} {header_border} text-align: {"left" if i==0 else "center"};">{col}</th>' for i, col in enumerate(display_cols))}
                </tr>
            </thead>
            <tbody>
    """
    
    current_group = None
    for row_idx, (_, row) in enumerate(df.iterrows()):
        # Handle Sub-headers
        if group_col and row[group_col] != current_group:
            current_group = row[group_col]
            html += f'<tr class="group-row"><td colspan="{len(display_cols)}" style="{group_header_style}">{current_group}</td></tr>'

        html += "<tr>"
        for col in display_cols:
            val_raw = row[col]
            
            # 6. Null Handling
            if pd.isna(val_raw) or val_raw is None:
                val = "—"
                is_number = False
                is_significant = False
            else:
                val = str(val_raw)
                is_number = False
                is_significant = False
                
                # Try numeric formatting
                try:
                    clean_val = val.replace(',', '').replace('%', '')
                    num_val = float(clean_val)
                    is_number = True
                    
                    # 1. Significance Check
                    is_q_col = any(x in col.lower() for x in ['q-value', 'p-value', 'q_value', 'p_value'])
                    if is_q_col:
                        if num_val < SIGNIFICANCE_THRESHOLD:
                            is_significant = True
                    
                    # 9. Unified 3 Sig-Fig Formatting
                    if is_q_col:
                        val = format_3_sig_figs(num_val)
                    elif 'pct' in col.lower() or '%' in col:
                        val = format_3_sig_figs(num_val) + "%"
                    elif any(x in col.lower() for x in ['enrich', 'effect', 'lift', 'ratio']):
                        val = format_3_sig_figs(num_val)
                    elif num_val == int(num_val) and 'N' in col:
                        val = f"{int(num_val):,}"
                    else:
                        val = format_3_sig_figs(num_val)
                except:
                    pass

            # 4. Professional Alignment
            # The first column (usually Characteristic) is left-aligned text. 
            # All other columns (usually values/ranges) are center-aligned to be consistent.
            if col == display_cols[0]:
                cell_class = "text-cell"
            else:
                cell_class = "center-cell"
                
            sig_class = "sig-value" if is_significant else ""
            
            # Handle Indentation for the first column
            row_style = base_cell
            if col == display_cols[0]:
                indent_level = 0
                
                # 1. Determine indentation level based on explicit tags
                if "[INDENT2]" in val:
                    indent_level = 2
                elif "[INDENT1]" in val:
                    indent_level = 1
                
                # 2. Special Case: Bold headers should always be flush left
                if "<b>" in val or "<strong>" in val:
                    indent_level = 0
                
                if indent_level > 0:
                    # Strip the tags and any extra &nbsp; for the display text
                    val = val.replace("[INDENT2]", "").replace("[INDENT1]", "").replace("&nbsp;", "").strip()
                    # Hierarchy: 0pt (Header) -> 20pt (Variable) -> 40pt (Subcategory)
                    padding_left = 20 if indent_level == 1 else 40
                    row_style += f" padding-left: {padding_left}pt;"
                else:
                    # Flush left for headers (ensure tags are removed if any)
                    val = val.replace("[INDENT2]", "").replace("[INDENT1]", "").strip()
                    row_style += " padding-left: 6pt;"

            # 160: Handle list formatting - replace semicolons with line breaks for better readability
            if ';' in val and len(val) > 15 and not is_number:
                val = val.replace('; ', '<br>')
            
            html += f'<td class="{cell_class} {sig_class}" style="{row_style}">{val}</td>'
        html += "</tr>"
    
    html += "</tbody></table>"
    
    # 3. Footnotes
    if footnotes:
        for fn in footnotes:
            html += f'<div class="footnote"><sup>*</sup>{fn}</div>'
            
    html += "</body></html>"
    
    output_path = output_dir / f"{filename}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

def main():
    """CLI tool for converting CSV to professional manuscript HTML table"""
    import argparse
    parser = argparse.ArgumentParser(description="Convert CSV to Professional Manuscript HTML Table")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("--title", "-t", help="Table title")
    parser.add_argument("--out", "-o", help="Output filename (without extension)")
    parser.add_argument("--dir", "-d", help="Output directory")
    parser.add_argument("--compact", "-c", action="store_true", help="Use compact mode")
    parser.add_argument("--number", "-n", help="Table number (e.g., '1', 'S2')")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        return
        
    df = pd.read_csv(input_path)
    title = args.title or input_path.stem.replace('_', ' ').title()
    filename = args.out or input_path.stem
    output_dir = Path(args.dir) if args.dir else TABLES_OUTPUT_DIR
    
    saved_path = save_manuscript_html(
        df, 
        title, 
        filename, 
        output_dir, 
        compact=args.compact, 
        table_number=args.number
    )
    print(f"Professional HTML table saved to: {saved_path}")

if __name__ == "__main__":
    main()
