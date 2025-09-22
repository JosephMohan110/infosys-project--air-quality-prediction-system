import shutil
from pathlib import Path

# project root (this file must be saved inside info-project/)
BASE_DIR = Path(__file__).resolve().parent

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# make sure correct dirs exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 1) Move enhanced_models if exists
enhanced_models = NOTEBOOKS_DIR / "data" / "enhanced_models"
if enhanced_models.exists():
    print(f"Moving {enhanced_models} -> {MODELS_DIR}")
    shutil.move(str(enhanced_models), str(MODELS_DIR / "enhanced_models"))

# 2) Move enhanced_results if exists
enhanced_results = NOTEBOOKS_DIR / "data" / "enhanced_results"
if enhanced_results.exists():
    print(f"Moving {enhanced_results} -> {RESULTS_DIR}")
    shutil.move(str(enhanced_results), str(RESULTS_DIR / "enhanced_results"))

# 3) Move notebooks/results -> results
wrong_results = NOTEBOOKS_DIR / "results"
if wrong_results.exists():
    print(f"Moving {wrong_results} -> {RESULTS_DIR}")
    for item in wrong_results.iterdir():
        shutil.move(str(item), str(RESULTS_DIR / item.name))
    wrong_results.rmdir()

# 4) Move notebooks/models -> models
wrong_models = NOTEBOOKS_DIR / "models"
if wrong_models.exists():
    print(f"Moving {wrong_models} -> {MODELS_DIR}")
    for item in wrong_models.iterdir():
        shutil.move(str(item), str(MODELS_DIR / item.name))
    wrong_models.rmdir()

# 5) Remove empty data folder
wrong_data = NOTEBOOKS_DIR / "data"
if wrong_data.exists() and not any(wrong_data.iterdir()):
    print(f"Removing empty folder {wrong_data}")
    wrong_data.rmdir()

print("✅ Folder structure corrected.")
