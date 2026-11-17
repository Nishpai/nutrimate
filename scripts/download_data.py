"""Download the raw source files into data/raw/.

You only need this if you want to rebuild data/processed/ yourself. The processed
subset used by the app is already included in the project.

Sources (all downloadable without an account):
  1. Food.com Recipes and Reviews (CC0, originally published on Kaggle by Alvin
     Irkaal). We use the copy inside the jiava/foodrs-data Hugging Face dataset,
     which stores the untouched recipes and reviews tables in a DuckDB file (~750 MB).
  2. Indian Nutrient Databank (INDB), Anuvaad, release 2024.11 (xlsx).
  3. USDA FoodData Central, SR Legacy (April 2018), CSV zip. Public domain (CC0).
"""

import zipfile

import _path  # noqa: F401
import requests

from nutrimate.config import DATA_RAW

FILES = {
    "foodrs.duckdb": "https://huggingface.co/datasets/jiava/foodrs-data/resolve/main/data/foodrs.duckdb",
    "Anuvaad_INDB_2024.11.xlsx": "https://www.anuvaad.org.in/wp-content/uploads/2020/07/Anuvaad_INDB_2024.11.xlsx",
    "FoodData_Central_sr_legacy_food_csv_2018-04.zip": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip",
}


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already have {dest.name}, skipping")
        return
    print(f"downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    print(f"  saved {dest.stat().st_size / 1e6:.1f} MB")


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(url, DATA_RAW / name)
    z = DATA_RAW / "FoodData_Central_sr_legacy_food_csv_2018-04.zip"
    with zipfile.ZipFile(z) as zf:
        zf.extractall(DATA_RAW / "usda_sr_legacy")
    print("done. Next: python scripts/inspect_data.py and python scripts/preprocess.py")


if __name__ == "__main__":
    main()
