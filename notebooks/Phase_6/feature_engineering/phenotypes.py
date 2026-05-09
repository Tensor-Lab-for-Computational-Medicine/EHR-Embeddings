import ast
import math
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd


def _read_rules_csv(csv_path: str) -> pd.DataFrame:
    # Robust parser: split on commas only at top-level (not inside quotes/brackets/parentheses)
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for i, raw in enumerate(f):
            line = raw.rstrip('\n').rstrip('\r')
            if not line or line.strip().startswith('#'):
                continue
            if i == 0 and line.lower().startswith('phenotype_name,'):
                continue
            fields = []
            buf = []
            depth_paren = depth_brack = depth_brace = 0
            in_sq = in_dq = False
            prev = ''
            for ch in line:
                if ch == '"' and not in_sq and prev != '\\':
                    in_dq = not in_dq
                elif ch == "'" and not in_dq and prev != '\\':
                    in_sq = not in_sq
                elif not in_sq and not in_dq:
                    if ch == '(':
                        depth_paren += 1
                    elif ch == ')':
                        depth_paren = max(0, depth_paren - 1)
                    elif ch == '[':
                        depth_brack += 1
                    elif ch == ']':
                        depth_brack = max(0, depth_brack - 1)
                    elif ch == '{':
                        depth_brace += 1
                    elif ch == '}':
                        depth_brace = max(0, depth_brace - 1)
                if ch == ',' and not in_sq and not in_dq and depth_paren == depth_brack == depth_brace == 0:
                    fields.append(''.join(buf).strip())
                    buf = []
                else:
                    buf.append(ch)
                prev = ch
            if buf:
                fields.append(''.join(buf).strip())
            if len(fields) < 4:
                continue
            name, ptype = fields[0], fields[1]
            logic = fields[2]
            required = fields[-1]
            rationale = ",".join(fields[3:-1]).strip() if len(fields) > 4 else ''
            if (len(logic) >= 2) and ((logic[0] == logic[-1]) and logic[0] in ('"', "'")):
                logic = logic[1:-1].strip()
            rows.append({
                'phenotype_name': name,
                'phenotype_type': ptype,
                'logic': logic,
                'rationale': rationale,
                'required_features': required,
            })
    df = pd.DataFrame(rows)
    for col in ['phenotype_name', 'phenotype_type', 'logic', 'rationale', 'required_features']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _ensure_required_columns(context_df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    # Create any missing required columns with safe defaults
    for name in required:
        if not name or name == 'nan':
            continue
        if name in context_df.columns:
            continue
        # Default for unknown required derived features: False (binary) or NaN otherwise
        # Since we don't have types here, default to False which also casts to 0 for numeric operations
        context_df[name] = False
    return context_df


def _eval_python_logic(expr: str, context_df: pd.DataFrame) -> pd.Series:
    # Evaluate a pythonic expression using df[...] and numpy
    local_env = {
        'df': context_df,
        'np': np,
        'pd': pd,
        'math': math,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
    }
    with np.errstate(all='ignore'):
        return eval(expr, {"__builtins__": {}}, local_env)


def _parse_case_when(case_expr: str, context_df: pd.DataFrame) -> pd.Series:
    # Robust CASE WHEN parser using regex; supports multiple WHEN ... THEN ...; optional ELSE ... END
    s = case_expr.strip()
    if not s.upper().startswith('CASE'):
        raise ValueError('Not a CASE WHEN expression')
    s = s[4:].strip()
    pattern = re.compile(r"\bWHEN\s+(.*?)\s+THEN\s+(.*?)(?=\bWHEN\b|\bELSE\b|\bEND\b)", re.IGNORECASE | re.DOTALL)
    parts: List[Tuple[str, str]] = [(m.group(1).strip(), m.group(2).strip()) for m in pattern.finditer(s)]
    default_value = None
    m_else = re.search(r"\bELSE\s+(.*?)\s*END\b", s, flags=re.IGNORECASE | re.DOTALL)
    if m_else:
        default_value = m_else.group(1).strip()

    # Evaluate conditions and choose values
    conditions = []
    values = []
    for cond_expr, raw_val in parts:
        mask = _eval_python_logic(cond_expr, context_df).astype(bool)
        val = raw_val
        if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
            val = val[1:-1]
        else:
            try:
                val = _eval_python_logic(val, context_df)
            except Exception:
                val = str(val)
        conditions.append(mask)
        values.append(val)

    if default_value is None:
        default_value = np.nan
    else:
        dv = default_value
        if (dv.startswith("'") and dv.endswith("'")) or (dv.startswith('"') and dv.endswith('"')):
            default_value = dv[1:-1]
        else:
            try:
                default_value = _eval_python_logic(dv, context_df)
            except Exception:
                default_value = str(dv)
    with np.errstate(all='ignore'):
        out = np.select(conditions, values, default=default_value)
    return pd.Series(out, index=context_df.index)


def compute_phenotypes(raw_df: pd.DataFrame, rules_csv_path: str) -> pd.DataFrame:
    """Compute phenotype features from raw numeric dataframe and a CSV rule set.

    raw_df is expected to be on original clinical scales (inverse-transformed).
    """
    rules = _read_rules_csv(rules_csv_path)
    context_df = raw_df.copy()
    for _, row in rules.iterrows():
        name = row['phenotype_name']
        ptype = row['phenotype_type'].lower().strip()
        logic = row['logic']
        required = []
        if 'required_features' in row and isinstance(row['required_features'], str):
            raw_req = row['required_features']
            # Some rows may accidentally include rationale content; keep only plausible feature tokens
            toks = [x.strip() for x in raw_req.split('|') if x and x.strip()]
            # Filter out tokens containing spaces likely from rationale sentences (heuristic)
            required = [t for t in toks if (' ' not in t) or t.endswith(('_mean_last','_mean_count','_stddev','_slope','_6h'))]
            # Expand special token [All _count features]
            if any(t.startswith('[') and '_count' in t for t in required):
                required = [t for t in required if '[' not in t]
                required += [c for c in context_df.columns if '_count' in c]
        try:
            if logic.strip().upper().startswith('CASE'):
                s = _parse_case_when(logic, context_df)
            else:
                s = _eval_python_logic(logic, context_df)
        except Exception as e:
            print(f"Phenotype rule failed [{name}]: {e}")
            # Fallback: fill with NaN/False to avoid breaking the pipeline
            if ptype == 'binary':
                s = pd.Series(False, index=context_df.index)
            else:
                s = pd.Series(np.nan, index=context_df.index)
        # Ensure we operate on a Series; broadcast scalars/arrays
        if not isinstance(s, pd.Series):
            if isinstance(s, np.ndarray):
                s = pd.Series(s, index=context_df.index)
            else:
                s = pd.Series([s] * len(context_df), index=context_df.index)
        # Enforce type
        if ptype == 'binary':
            s = s.astype(bool)
        elif ptype == 'continuous':
            s = pd.to_numeric(s, errors='coerce')
        elif ptype == 'categorical':
            s = s.astype(str)
        else:
            # Default to string
            s = s.astype(str)
        context_df[name] = s
    # Return only engineered phenotype columns (in rule order)
    cols = rules['phenotype_name'].tolist()
    return context_df[cols]


def compute_and_save_phenotypes(x_test_path: str, scaler_path: str, rules_csv_path: str, out_path: str) -> str:
    # Load X_test and inverse-transform to original scale where applicable
    with open(x_test_path, 'rb') as f:
        X_test = pd.read_pickle(f) if str(x_test_path).lower().endswith('.pkl') else None
    if X_test is None:
        X_test = pd.read_pickle(x_test_path)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    # Inverse transform using saved scaler if available
    try:
        import pickle
        with open(scaler_path, 'rb') as f:
            sc = pickle.load(f)
        if hasattr(sc, 'feature_names_in_'):
            X_test[sc.feature_names_in_] = sc.inverse_transform(X_test[sc.feature_names_in_])
        else:
            X_test = pd.DataFrame(sc.inverse_transform(X_test), columns=X_test.columns, index=X_test.index)
    except Exception:
        pass
    phenos = compute_phenotypes(X_test, rules_csv_path)
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    phenos.to_pickle(out_path)
    # Also write CSV alongside for inspection
    try:
        phenos.to_csv(out_path.replace('.pkl', '.csv'), index=False)
    except Exception:
        pass
    return out_path


