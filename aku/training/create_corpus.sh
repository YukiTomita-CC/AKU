#!/bin/bash

DIR_PATH="aku/training/dataset/raw"
train_file="aku/training/dataset/train.txt"
test_file="aku/training/dataset/test.txt"

> "$train_file"
> "$test_file"

for file in "$DIR_PATH"/*.txt; do
  echo "Processing $file ..."

  line_count=$(wc -l < "$file")

  train_lines=$(( line_count * 95 / 100 ))
  test_lines=$(( line_count - train_lines ))

  head -n "$train_lines" "$file" >> "$train_file"
  tail -n "$test_lines" "$file" >> "$test_file"

done

echo "Processing has been completed."
