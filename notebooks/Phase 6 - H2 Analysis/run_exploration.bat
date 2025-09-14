@echo off
setlocal enabledelayedexpansion

REM Activate conda environment
CALL conda activate mimic-legacy

REM Move to script directory
cd /d "%~dp0"

REM Build phenotypes once (if needed)
if not exist "feature_engineering\artifacts\X_test_phenotypes.pkl" (
    echo === Building phenotypes ===
    python feature_engineering\build_phenotypes.py ^
      --x_test_path "..\Phase 1 and 2\phase_1_outputs\X_test.pkl" ^
      --scaler_path "..\Phase 1 and 2\phase_1_outputs\scaler.pkl" ^
      --rules_csv "feature_engineering\feature_rules.csv" ^
      --out_path "feature_engineering\artifacts\X_test_phenotypes.pkl"
)

REM Depth sweep to use for all phases
set DEPTHS=2,3,4,5

REM Run full pipeline for a given config
REM %1 is the config file (should be in h2b/ directory)
set CFG=%1
if "%CFG%"=="" set CFG=h2b\config_h2_morthosp.py

echo === Running Phase IV (H2b) for %CFG% ===
python h2b\h2b_analysis.py --config_file %CFG% --depths %DEPTHS%

echo === Running Phase IV-B (Direct Discordance) for %CFG% ===
python h2b\h2b_direct_discordance.py --config_file %CFG% --depths %DEPTHS%

echo === Running cross-depth archetype selector for Phase IV ===
python h2b\cross_depth_archetype_selector.py --config_file %CFG% --phase IV --min_coverage 25 --min_lift 1.5 --jaccard_thresh 0.60 --max_archetypes 3

echo === Running cross-depth archetype selector for Phase IV-B ===
python h2b\cross_depth_archetype_selector.py --config_file %CFG% --phase IVB --min_coverage 25 --min_lift 1.5 --jaccard_thresh 0.60 --max_archetypes 3

echo === Running Phase V Meta (Phase IV) for %CFG% ===
python h2b\h2v_meta_analysis.py --config_file %CFG% --phase IV

echo === Running Phase V Meta (Phase IV-B) for %CFG% ===
python h2b\h2v_meta_analysis.py --config_file %CFG% --phase IVB

echo --- COMPLETE for %CFG% ---

REM If no second arg provided, also run for readmission config
if not "%2"=="skip_readmin" (
  echo === Running full pipeline for h2b\config_h2_readmin30.py ===
  python h2b\h2b_analysis.py --config_file h2b\config_h2_readmin30.py --depths %DEPTHS%
  python h2b\h2b_direct_discordance.py --config_file h2b\config_h2_readmin30.py --depths %DEPTHS%
  echo === Running cross-depth archetype selector for Phase IV (readmission) ===
  python h2b\cross_depth_archetype_selector.py --config_file h2b\config_h2_readmin30.py --phase IV --min_coverage 25 --min_lift 1.5 --jaccard_thresh 0.60 --max_archetypes 3
  echo === Running cross-depth archetype selector for Phase IV-B (readmission) ===
  python h2b\cross_depth_archetype_selector.py --config_file h2b\config_h2_readmin30.py --phase IVB --min_coverage 25 --min_lift 1.5 --jaccard_thresh 0.60 --max_archetypes 3
  python h2b\h2v_meta_analysis.py --config_file h2b\config_h2_readmin30.py --phase IV
  python h2b\h2v_meta_analysis.py --config_file h2b\config_h2_readmin30.py --phase IVB
  echo --- COMPLETE for h2b\config_h2_readmin30.py ---
)

echo === ALL RUNS COMPLETE ===

endlocal
