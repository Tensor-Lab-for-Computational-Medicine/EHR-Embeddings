import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path for utility imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from manuscript_table_utils import save_manuscript_html, TABLES_OUTPUT_DIR

def debug_indentation():
    # Create a simple test dataframe with explicit tags
    data = [
        {'Characteristic': '<b>Demographics</b>', 'Value': ''},
        {'Characteristic': '[INDENT1]Age, median [IQR], y', 'Value': '65.2'},
        {'Characteristic': '[INDENT1]Sex, No.', 'Value': ''},
        {'Characteristic': '[INDENT2]Female', 'Value': '9,661'},
        {'Characteristic': '[INDENT1]Race and Ethnicity, No.', 'Value': ''},
        {'Characteristic': '[INDENT2]White', 'Value': '15,908'},
        {'Characteristic': '[INDENT2]Black', 'Value': '1,773'},
    ]
    df = pd.DataFrame(data)
    
    output_dir = TABLES_OUTPUT_DIR
    html_path = save_manuscript_html(
        df, 
        "Debug Indentation Table", 
        "Table_Debug_Indentation", 
        output_dir
    )
    print(f"Debug table saved to: {html_path}")
    
    # Check the generated HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Check for specific padding values
    checks = {
        'Demographics': 'padding-left: 6pt',
        'Age': 'padding-left: 20pt',
        'Female': 'padding-left: 40pt',
        'White': 'padding-left: 40pt'
    }
    
    print("\nVerification Results:")
    for label, expected in checks.items():
        found = expected in content
        print(f"Checking '{label}' for '{expected}': {'PASS' if found else 'FAIL'}")
        if not found:
            # Find the line for context
            lines = content.split('<tr>')
            for line in lines:
                if label in line:
                    print(f"  Actual line: {line[:200]}...")

if __name__ == "__main__":
    debug_indentation()
