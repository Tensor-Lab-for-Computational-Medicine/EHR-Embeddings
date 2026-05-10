@echo off
setlocal enabledelayedexpansion

REM Use explicit python interpreter to avoid environment activation issues
set PYTHON_EXE=D:\Programs\Miniconda3\envs\mimic-legacy\python.exe

REM Move to script directory
cd /d "%~dp0"

REM Build phenotypes for test/train once (if needed)
if not exist "feature_engineering\artifacts\X_test_phenotypes.pkl" (
    echo === Building phenotypes (test) ===
    %PYTHON_EXE% feature_engineering\build_phenotypes.py ^
      --x_path "..\Phase_1-2\phase_1_outputs\X_test.pkl" ^
      --scaler_path "..\Phase_1-2\phase_1_outputs\scaler.pkl" ^
      --rules_csv "feature_engineering\feature_rules.csv" ^
      --out_path "feature_engineering\artifacts\X_test_phenotypes.pkl"
)
if not exist "feature_engineering\artifacts\X_trainval_phenotypes.pkl" (
    echo === Building phenotypes (train+val) ===
    %PYTHON_EXE% feature_engineering\build_phenotypes.py ^
      --x_path "..\Phase_1-2\phase_1_outputs\X_train.pkl" ^
      --x2_path "..\Phase_1-2\phase_1_outputs\X_val.pkl" ^
      --scaler_path "..\Phase_1-2\phase_1_outputs\scaler.pkl" ^
      --rules_csv "feature_engineering\feature_rules.csv" ^
      --out_path "feature_engineering\artifacts\X_trainval_phenotypes.pkl"
)

REM Depth sweep to use for all phases
set DEPTHS=2,3,4,5,6,7

REM Run full pipeline for a given config
REM %1 is the config file (should be in h2b/ directory)
set CFG=%1
if "%CFG%"=="" set CFG=h2b\config_h2_morthosp.py

REM Resolve H2a module name from CFG path
set H2A_CFG=config_h2_morthosp
echo %CFG% | findstr /I "readmin" >nul
if %ERRORLEVEL% EQU 0 (
    set H2A_CFG=config_h2_readmin30
)

echo === Running H2a for %H2A_CFG% ===
%PYTHON_EXE% h2a\h2a_analysis.py --config %H2A_CFG%

echo === Running Phase IV + IV-B Combined for %CFG% ===
%PYTHON_EXE% h2b\h2b_combined.py --config_file %CFG% --depths %DEPTHS%

echo === Running cross-depth archetype selector for Phase IV ===
%PYTHON_EXE% h2b\cross_depth_archetype_selector.py --config_file %CFG% --phase IV --min_coverage 75 --min_lift 1.5 --jaccard_thresh 0.75 --max_archetypes 8

echo === Running cross-depth archetype selector for Phase IV-B ===
%PYTHON_EXE% h2b\cross_depth_archetype_selector.py --config_file %CFG% --phase IVB --min_coverage 75 --min_lift 1.5 --jaccard_thresh 0.75 --max_archetypes 8

REM Resolve final archetype paths for meta analysis
set RESULTS_DIR=h2b\h2_results\mort_hosp
echo %CFG% | findstr /I "readmin" >nul
if %ERRORLEVEL% EQU 0 (
    set RESULTS_DIR=h2b\h2_results\readmission_30
)

echo === Running Phase V Meta (Phase IV) for %CFG% ===
%PYTHON_EXE% h2b\h2v_meta_analysis.py --config_file %CFG% --phase IV --final_archetypes_path "%RESULTS_DIR%\final_archetypes.csv" --phenotypes_test_path "feature_engineering\artifacts\X_test_phenotypes.pkl" --scaler_path "..\Phase_1-2\phase_1_outputs\scaler.pkl"

echo === Running Phase V Meta (Phase IV-B) for %CFG% ===
%PYTHON_EXE% h2b\h2v_meta_analysis.py --config_file %CFG% --phase IVB --final_archetypes_path "%RESULTS_DIR%\final_archetypes_ivb.csv" --phenotypes_test_path "feature_engineering\artifacts\X_test_phenotypes.pkl" --scaler_path "..\Phase_1-2\phase_1_outputs\scaler.pkl"

echo --- COMPLETE for %CFG% ---

REM If no second arg provided, also run for readmission config
if not "%2"=="skip_readmin" (
  if not "%H2A_CFG%"=="config_h2_readmin30" (
      echo === Running H2a for readmission ===
      %PYTHON_EXE% h2a\h2a_analysis.py --config config_h2_readmin30
      echo === Running full pipeline for h2b\config_h2_readmin30.py ===
      %PYTHON_EXE% h2b\h2b_combined.py --config_file h2b\config_h2_readmin30.py --depths %DEPTHS%
      echo === Running cross-depth archetype selector for Phase IV (readmission) ===
      %PYTHON_EXE% h2b\cross_depth_archetype_selector.py --config_file h2b\config_h2_readmin30.py --phase IV --min_coverage 75 --min_lift 1.5 --jaccard_thresh 0.75 --max_archetypes 8
      echo === Running cross-depth archetype selector for Phase IV-B (readmission) ===
      %PYTHON_EXE% h2b\cross_depth_archetype_selector.py --config_file h2b\config_h2_readmin30.py --phase IVB --min_coverage 75 --min_lift 1.5 --jaccard_thresh 0.75 --max_archetypes 8
      %PYTHON_EXE% h2b\h2v_meta_analysis.py --config_file h2b\config_h2_readmin30.py --phase IV --final_archetypes_path "h2b\h2_results\readmission_30\final_archetypes.csv" --phenotypes_test_path "feature_engineering\artifacts\X_test_phenotypes.pkl" --scaler_path "..\Phase_1-2\phase_1_outputs\scaler.pkl"
      %PYTHON_EXE% h2b\h2v_meta_analysis.py --config_file h2b\config_h2_readmin30.py --phase IVB --final_archetypes_path "h2b\h2_results\readmission_30\final_archetypes_ivb.csv" --phenotypes_test_path "feature_engineering\artifacts\X_test_phenotypes.pkl" --scaler_path "..\Phase_1-2\phase_1_outputs\scaler.pkl"
      echo --- COMPLETE for h2b\config_h2_readmin30.py ---
  )
)

echo === ALL RUNS COMPLETE ===

echo === Running Bias Validation (Sensitivity Analysis) ===
%PYTHON_EXE% h2b\sensitivity_analysis.py

echo === Generating Meta-Feature Forest Plots ===
%PYTHON_EXE% h2b\plot_meta_features.py

echo === Generating Phase V Meta Tables (LaTeX) ===
%PYTHON_EXE% h2b\generate_phase_v_meta_tables.py

echo === Generating Archetype Reports (Cards ^& Tables) ===
%PYTHON_EXE% h2b\generate_archetype_reports.py

echo === Generating Manuscript-Ready Table (Google Docs) ===
%PYTHON_EXE% h2b\generate_manuscript_table.py

endlocal
