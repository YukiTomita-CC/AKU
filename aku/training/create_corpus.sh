#!/bin/bash

if [ ! -d "aku" ]; then
  echo "This script must be run from the AKU directory."
  exit 1
fi

mkdir -p "aku/training/prepare/processed_data"

DIR_PATH="aku/training/prepare/raw_data"
train_file="aku/training/prepare/processed_data/train.txt"
test_file="aku/training/prepare/processed_data/test.txt"

if [ -f "$train_file" ] && [ -f "$test_file" ]; then
  echo "train.txt and test.txt already exist. Processing has been completed."
  exit 0
fi

> "$train_file"
> "$test_file"

for file in "$DIR_PATH"/*.txt; do
  line_count=$(wc -l < "$file")

  train_lines=$(( line_count * 95 / 100 ))
  test_lines=$(( line_count - train_lines ))

  head -n "$train_lines" "$file" >> "$train_file"
  tail -n "$test_lines" "$file" >> "$test_file"

done

echo "Processing has been completed."
