#!/usr/bin/env bash
# Download the ESCI parquet files (git-lfs objects) into data/raw/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
base="https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset"
for f in shopping_queries_dataset_examples.parquet shopping_queries_dataset_products.parquet; do
  [ -s "data/raw/$f" ] && { echo "have $f"; continue; }
  echo "downloading $f"; curl -L --progress-bar -o "data/raw/$f" "$base/$f"
done
ls -la data/raw
