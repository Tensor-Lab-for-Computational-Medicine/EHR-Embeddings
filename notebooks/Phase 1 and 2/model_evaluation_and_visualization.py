import os
import pickle
import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, brier_score_loss
)
import shap


# -----------------------------------------------------------------------------
# Paths & setup
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_1_DIR = os.path.join(BASE_DIR, 'phase_1_outputs')
EVALUATION_DIR = os.path.join(BASE_DIR, 'model_evaluation_outputs')
os.makedirs(EVALUATION_DIR, exist_ok=True)

plt.style.use('default')
sns.set_palette('husl')
FIGURE_SIZE = (12, 8)
DPI = 300

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(EVALUATION_DIR, 'evaluation_log.txt'), mode='w'),
        logging.StreamHandler()
    ]
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def resolve_in_dirs(filename: str, candidate_dirs: List[str]) -> Optional[str]:
    for d in candidate_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


def load_models_and_results(preferred_dir: Optional[str] = None, allow_fallback: bool = True) -> Tuple[Dict, Dict]:
    """Load available baseline models (XGBoost, Elastic Net) and stored results.
    If preferred_dir is provided, search it first. When allow_fallback is False,
    only search the preferred_dir.
    """
    candidates: List[str] = []
    if preferred_dir:
        # allow absolute or relative to PHASE_1_DIR
        pref_abs = preferred_dir if os.path.isabs(preferred_dir) else os.path.join(PHASE_1_DIR, preferred_dir)
        if os.path.isdir(pref_abs):
            candidates.append(pref_abs)
        else:
            logging.warning(f"Preferred model dir not found: {pref_abs}")
            if not allow_fallback:
                raise FileNotFoundError(f"Preferred model dir not found: {pref_abs}")

    if allow_fallback or not candidates:
        # Broader search order
        extra = [
            PHASE_1_DIR,
            os.path.join(PHASE_1_DIR, 'mort_hosp'),
            os.path.join(PHASE_1_DIR, 'los_3'),
            os.path.join(PHASE_1_DIR, 'los_7'),
            os.path.join(PHASE_1_DIR, 'readmission_30'),
            os.path.join(PHASE_1_DIR, 'intervention_vaso'),
            os.path.join(PHASE_1_DIR, 'intervention_vent'),
        ]
        for d in extra:
            if d not in candidates:
                candidates.append(d)

    files = {
        'XGBoost': {
            'model': 'model_1_xgboost_baseline.pkl',
            'results': 'results_xgboost_baseline.pkl',
            'color': '#1f77b4',
            'linestyle': '-'
        },
        'Elastic Net': {
            'model': 'model_2_elastic_net_baseline.pkl',
            'results': 'results_elastic_net_baseline.pkl',
            'color': '#ff7f0e',
            'linestyle': '--'
        },
    }

    models: Dict[str, Any] = {}
    results: Dict[str, Dict] = {}

    for name, meta in files.items():
        model_path = resolve_in_dirs(meta['model'], candidates)
        res_path = resolve_in_dirs(meta['results'], candidates)

        if not model_path:
            logging.warning(f"Model file not found for {name}: {meta['model']} in {candidates}")
            continue

        with open(model_path, 'rb') as f:
            models[name] = pickle.load(f)
        logging.info(f"Loaded {name} model from {os.path.relpath(model_path, BASE_DIR)}")

        if res_path and os.path.exists(res_path):
            with open(res_path, 'rb') as f:
                try:
                    results[name] = pickle.load(f)
                except Exception:
                    results[name] = {}
            logging.info(f"Loaded {name} results from {os.path.relpath(res_path, BASE_DIR)}")
        else:
            results[name] = {}

        # attach style
        results[name]['color'] = meta['color']
        results[name]['linestyle'] = meta['linestyle']

    return models, results


def load_test_data(target_column: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Load aligned test features/labels from pickles in phase_1_outputs.
    Selects a primary target column if y is multi-label and sanitizes values.
    If target_column is provided, it is used when present; otherwise auto-selected.
    """
    X_path = os.path.join(PHASE_1_DIR, 'X_test.pkl')
    y_path = os.path.join(PHASE_1_DIR, 'y_test.pkl')
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        logging.error('Missing X_test.pkl or y_test.pkl under phase_1_outputs')
        return None, None

    X_test = pd.read_pickle(X_path)
    y_obj = pd.read_pickle(y_path)

    # choose target column if DataFrame
    def _select_target(y_any: Any) -> pd.Series:
        if isinstance(y_any, pd.Series):
            return y_any
        if isinstance(y_any, pd.DataFrame):
            if target_column is not None:
                if target_column in y_any.columns:
                    logging.info(f"Using specified target column: {target_column}")
                    return y_any[target_column]
                else:
                    raise ValueError(f"Requested target column '{target_column}' not found. Available: {list(y_any.columns)[:10]}...")
            preferred = [
                'mort_hosp', 'mortality_hosp', 'mort_icu', 'in_hospital_mortality', 'in_hosp_mort',
                'readmission_30', 'intervention_vaso', 'intervention_vent', 'los_3', 'los_7',
            ]
            for col in preferred:
                if col in y_any.columns:
                    return y_any[col]
            # pick first column with <=2 unique non-null values (likely binary)
            for col in y_any.columns:
                vals = y_any[col].dropna().unique()
                if len(vals) <= 2:
                    return y_any[col]
            return y_any.iloc[:, 0]
        return pd.Series(y_any)

    def _coerce_binary(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s.astype(int)
        if s.dtype == object:
            mapping = {
                'yes': 1, 'no': 0, 'true': 1, 'false': 0,
                'died': 1, 'survived': 0, 'death': 1, 'alive': 0
            }
            s = s.astype(str).str.lower().map(mapping)
        return pd.to_numeric(s, errors='coerce')

    y_series = _coerce_binary(_select_target(y_obj)).squeeze()

    # align indices
    common_idx = X_test.index.intersection(y_series.index)
    X_test = X_test.loc[common_idx]
    y_series = y_series.loc[common_idx]
    # remove non-finite
    y_series = y_series.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.loc[y_series.index]

    logging.info(f"Loaded test data: X_test{X_test.shape}, y_test{y_series.shape}")
    try:
        uniq = pd.unique(y_series)
        logging.info(f"y_test unique values (up to 5): {list(uniq[:5])}")
    except Exception:
        pass
    return X_test, y_series


def generate_predictions(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    preds: Dict[str, Dict] = {}
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            raw = model.decision_function(X_test)
            proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

        proba = np.asarray(proba, dtype=float)
        finite = np.isfinite(proba)
        y_mask = ~y_test.isna().values
        mask = finite & y_mask
        if mask.sum() < len(proba):
            logging.warning(f"Dropping {len(proba)-mask.sum()} samples due to NaN/inf in predictions or labels")
        Xm = X_test.iloc[mask]
        yt = y_test.values[mask]
        yh = model.predict(Xm)

        preds[name] = {
            'y_true': yt,
            'y_pred_proba': proba[mask],
            'y_pred': yh,
            'auroc': roc_auc_score(yt, proba[mask]),
            'auprc': average_precision_score(yt, proba[mask]),
            'classification_report': classification_report(yt, yh, output_dict=True),
            'confusion_matrix': confusion_matrix(yt, yh),
        }
        logging.info(f"{name}: AUROC={preds[name]['auroc']:.4f} AUPRC={preds[name]['auprc']:.4f}")
    return preds


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_roc_curves(preds: Dict, save_path: Optional[str] = None):
    plt.figure(figsize=FIGURE_SIZE)
    for name, d in preds.items():
        fpr, tpr, _ = roc_curve(d['y_true'], d['y_pred_proba'])
        color = d.get('color') or '#1f77b4'
        linestyle = d.get('linestyle') or '-'
        plt.plot(fpr, tpr, label=f"{name} (AUROC={d['auroc']:.3f})", color=color, linestyle=linestyle, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlim(0, 1); plt.ylim(0, 1.05)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves'); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    if save_path: plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()


def plot_pr_curves(preds: Dict, y_test: pd.Series, save_path: Optional[str] = None):
    plt.figure(figsize=FIGURE_SIZE)
    baseline = y_test.mean()
    for name, d in preds.items():
        precision, recall, _ = precision_recall_curve(d['y_true'], d['y_pred_proba'])
        color = d.get('color') or '#1f77b4'
        linestyle = d.get('linestyle') or '-'
        plt.plot(recall, precision, label=f"{name} (AUPRC={d['auprc']:.3f})", color=color, linestyle=linestyle, linewidth=2)
    plt.axhline(baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline={baseline:.3f}')
    plt.xlim(0, 1); plt.ylim(0, 1.05)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curves'); plt.legend(loc='lower left'); plt.grid(True, alpha=0.3)
    if save_path: plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(preds: Dict, save_path: Optional[str] = None):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    axes = [axes] if n == 1 else axes
    for ax, (name, d) in zip(axes, preds.items()):
        cm = d['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Survived', 'Died'], yticklabels=['Survived', 'Died'], ax=ax)
        ax.set_title(f'{name}\nConfusion Matrix')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()


def calculate_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    fracs, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.sum() > 0:
            fracs.append(y_true[m].mean())
            means.append(y_prob[m].mean())
    return np.array(fracs), np.array(means)


def assess_model_calibration(preds: Dict, y_test: pd.Series, save_base: Optional[str] = None) -> Dict:
    n = len(preds)
    fig, axes = plt.subplots(n, 1, figsize=(10, 6*n))
    axes = [axes] if n == 1 else axes
    out: Dict[str, Dict] = {}
    for ax, (name, d) in zip(axes, preds.items()):
        proba = d['y_pred_proba']
        yt = d['y_true']
        fop, mpv = calculate_calibration_curve(yt, proba, n_bins=10)
        brier = brier_score_loss(yt, proba)
        ax.plot(mpv, fop, 's-', label=f"{name} (Brier={brier:.4f})", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k:', label='Perfect')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{name} - Calibration')
        ax.legend(); ax.grid(True, alpha=0.3)
        out[name] = {'brier_score': brier}
    plt.tight_layout()
    if save_base:
        path = save_base.replace('.png', '_calibration.png')
        plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.show()
    return out


def create_feature_name_mapping(cols: List[str]) -> Dict[str, str]:
    abbr = {
        'alanine aminotransferase': 'ALT', 'asparate aminotransferase': 'AST', 'alkaline phosphate': 'ALP',
        'blood urea nitrogen': 'BUN', 'glascow coma scale': 'GCS', 'respiratory rate': 'RR', 'heart rate': 'HR',
        'blood pressure': 'BP', 'systolic blood pressure': 'SBP', 'diastolic blood pressure': 'DBP',
        'mean blood pressure': 'MAP', 'oxygen saturation': 'SpO2', 'partial pressure': 'P', 'temperature': 'Temp',
        'cardiac output': 'CO', 'central venous pressure': 'CVP', 'pulmonary artery pressure': 'PAP',
        'inspired oxygen': 'FiO2', 'positive end expiratory pressure': 'PEEP', 'tidal volume': 'TV',
        'glascow coma scale total': 'GCS Total', 'glascow coma scale motor response': 'GCS Motor',
        'glascow coma scale verbal response': 'GCS Verbal', 'glascow coma scale eye opening': 'GCS Eyes',
        '_mean': ' (avg)', '_std': ' (var)', '_slope_6h': ' (6h trend)', '_slope_24h': ' (24h trend)', '_encoded': ''
    }
    m: dict[str, str] = {}
    for c in cols:
        s = c.lower()
        for k, v in abbr.items():
            s = s.replace(k, v)
        s = s.replace('  ', ' ').strip()
        if len(s) > 25:
            parts = s.split()
            s = (' '.join(parts[:2]) + '...') if len(parts) > 1 else (s[:22] + '...')
        m[c] = s.capitalize()
    return m


def analyze_with_shap(models: Dict, X: pd.DataFrame, y: pd.Series, save_base: Optional[str] = None,
                      n_samples: int = 30000, include_interactions: bool = False) -> Dict:
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
    else:
        Xs, ys = X, y

    name_map = create_feature_name_mapping(Xs.columns.tolist())
    Xd = Xs.copy(); Xd.columns = [name_map[c] for c in Xd.columns]

    data: Dict[str, Dict] = {}
    for name, model in models.items():
        try:
            if hasattr(model, 'feature_importances_'):
                expl = shap.TreeExplainer(model)
                sv = expl.shap_values(Xs)
            elif hasattr(model, 'coef_'):
                expl = shap.LinearExplainer(model, Xs)
                sv = expl.shap_values(Xs)
            else:
                expl = shap.Explainer(model, Xs)
                res = expl(Xs)
                sv = res.values if hasattr(res, 'values') else res
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]
            data[name] = {'explainer': expl, 'shap_values': sv, 'X': Xs, 'Xd': Xd, 'y': ys}
            logging.info(f"SHAP computed for {name}")
        except Exception as e:
            logging.warning(f"SHAP failed for {name}: {e}")

    if not data:
        return {}

    # Summary (dot)
    n = len(data)
    fig, axes = plt.subplots(n, 1, figsize=(14, 8*n))
    axes = [axes] if n == 1 else axes
    for ax, (name, d) in zip(axes, data.items()):
        plt.sca(ax)
        shap.summary_plot(d['shap_values'], d['Xd'], plot_type='dot', show=False, max_display=20)
        ax.set_title(f'{name} - Feature Impact')
    plt.tight_layout()
    if save_base:
        p = save_base.replace('.png', '_summary.png')
        plt.savefig(p, dpi=DPI, bbox_inches='tight')
    plt.show()

    # Importance (bar)
    fig, axes = plt.subplots(n, 1, figsize=(12, 6*n))
    axes = [axes] if n == 1 else axes
    for ax, (name, d) in zip(axes, data.items()):
        plt.sca(ax)
        shap.summary_plot(d['shap_values'], d['Xd'], plot_type='bar', show=False, max_display=20)
        ax.set_title(f'{name} - Feature Importance')
    plt.tight_layout()
    if save_base:
        p = save_base.replace('.png', '_importance.png')
        plt.savefig(p, dpi=DPI, bbox_inches='tight')
    plt.show()

    # Waterfall: one survivor and one deceased if available
    survivors = d['y'][d['y'] == 0].index if len(data) else []
    deceased = d['y'][d['y'] == 1].index if len(data) else []
    pick_idx = []
    if len(survivors) > 0:
        s_idx = np.random.choice(survivors)
        pick_idx.append(('Survived', s_idx))
    if len(deceased) > 0:
        d_idx = np.random.choice(deceased)
        pick_idx.append(('Died', d_idx))

    for name, d in data.items():
        for outcome, ridx in pick_idx:
            try:
                pos = d['y'].index.get_loc(ridx)
                base = d['explainer'].expected_value
                if isinstance(base, np.ndarray):
                    base = base[1] if len(base) > 1 else base[0]
                vals = d['shap_values'][pos]
                row = d['Xd'].iloc[pos]
                shap.waterfall_plot(
                    shap.Explanation(values=vals, base_values=base, data=row.values, feature_names=row.index.tolist()),
                    max_display=15, show=False
                )
                # Title & save
                if hasattr(models[name], 'predict_proba'):
                    pred = models[name].predict_proba(d['X'].iloc[pos:pos+1])[:, 1][0]
                else:
                    pred = float(models[name].predict(d['X'].iloc[pos:pos+1])[0])
                plt.title(f'{name} - {outcome} Patient\nScore: {pred:.3f} (uncalibrated)')
                if save_base:
                    fn = f"_waterfall_{name.lower().replace(' ', '_')}_{outcome.lower()}_patient.png"
                    p = save_base.replace('.png', fn)
                    plt.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.show(); plt.close()
            except Exception as e:
                logging.warning(f"Waterfall failed for {name} ({outcome}): {e}")

    # Optional interactions for tree models
    if include_interactions:
        for name, d in data.items():
            if not hasattr(models[name], 'feature_importances_'):
                continue
            try:
                k = min(500, len(d['X']))
                idx = np.random.choice(len(d['X']), k, replace=False)
                Xi, Xdi = d['X'].iloc[idx], d['Xd'].iloc[idx]
                inter = d['explainer'].shap_interaction_values(Xi)
                fig = plt.figure(figsize=(14, 10))
                shap.summary_plot(inter, Xdi, plot_type='dot', show=False, max_display=15)
                plt.title(f'{name} - SHAP Interaction Effects')
                if save_base:
                    p = save_base.replace('.png', f"_interaction_summary_{name.lower().replace(' ', '_')}.png")
                    plt.savefig(p, dpi=DPI, bbox_inches='tight')
                plt.show(); plt.close()
            except Exception as e:
                logging.warning(f"Interaction analysis failed for {name}: {e}")

    return data


def create_performance_summary(preds: Dict, stored: Dict) -> pd.DataFrame:
    rows = []
    for name, d in preds.items():
        r = {
            'Model': name,
            'AUROC': d['auroc'],
            'AUPRC': d['auprc'],
            'Accuracy': d['classification_report']['accuracy'],
            'Precision (Class 1)': d['classification_report']['1']['precision'],
            'Recall (Class 1)': d['classification_report']['1']['recall'],
            'F1-Score (Class 1)': d['classification_report']['1']['f1-score'],
            'Specificity': d['classification_report']['0']['recall'],
        }
        sr = stored.get(name, {})
        for k in ['test_auroc_ci_lower','test_auroc_ci_upper','test_auroc_std','test_auprc_ci_lower','test_auprc_ci_upper','test_auprc_std']:
            if k in sr: r[k.replace('test_', '').upper()] = sr[k]
        rows.append(r)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Model evaluation and visualization')
    parser.add_argument('--target-col', type=str, default=None, help='Target column name in y_test.pkl (if DataFrame)')
    parser.add_argument('--model-dir', type=str, default=None, help='Preferred subdirectory under phase_1_outputs for models (e.g., mort_hosp)')
    parser.add_argument('--no-fallback', action='store_true', help='Only use the preferred model directory; do not fall back')
    args = parser.parse_args()

    logging.info('Starting evaluation...')
    models, stored = load_models_and_results(preferred_dir=args.model_dir, allow_fallback=not args.no_fallback)
    X_test, y_test = load_test_data(target_column=args.target_col)
    if not models or X_test is None:
        logging.error('Missing models or test data. Exiting.')
        return

    # attach styles to predictions
    preds = generate_predictions(models, X_test, y_test)
    for name in preds:
        preds[name]['color'] = stored.get(name, {}).get('color')
        preds[name]['linestyle'] = stored.get(name, {}).get('linestyle')

    plot_roc_curves(preds, os.path.join(EVALUATION_DIR, 'roc_curves.png'))
    plot_pr_curves(preds, y_test, os.path.join(EVALUATION_DIR, 'pr_curves.png'))
    plot_confusion_matrices(preds, os.path.join(EVALUATION_DIR, 'confusion_matrices.png'))
    assess_model_calibration(preds, y_test, os.path.join(EVALUATION_DIR, 'calibration_analysis.png'))

    shap_data = analyze_with_shap(models, X_test, y_test, os.path.join(EVALUATION_DIR, 'shap_analysis.png'))

    df = create_performance_summary(preds, stored)
    df.to_csv(os.path.join(EVALUATION_DIR, 'performance_summary.csv'), index=False)

    if shap_data:
        with open(os.path.join(EVALUATION_DIR, 'shap_data.pkl'), 'wb') as f:
            pickle.dump(shap_data, f)

    logging.info(f"Done. Outputs in {os.path.relpath(EVALUATION_DIR, BASE_DIR)}/")
    print(df.to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()


