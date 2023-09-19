#!/bin/bash

# Define the test files
test_files=("test_prepare.py" "test_model_components.py" "test_causal_attention.py" "test_gpt.py" "test_dataloader.py")

# Loop through the test files and run them using unittest
for file in "${test_files[@]}"; do
  echo "Running tests in $file";
  python -m unittest "$file";
done