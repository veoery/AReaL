#!/bin/bash
# Simple script to create index files for OS interaction training

cd /home/haizhonz/yizhuod/AReaL/examples/os_interaction/data

echo "Creating training indices (datasets 1-6)..."
> train_indices.jsonl

# Dataset 1
echo '{"index": "train-001-stock-00000", "messages": []}' >> train_indices.jsonl

# Dataset 2
echo '{"index": "train-002-environment-00000", "messages": []}' >> train_indices.jsonl

# Dataset 3
echo '{"index": "train-003-ac-00000", "messages": []}' >> train_indices.jsonl

# Dataset 4 (multiple files)
for file in N11 N225 N37 N4 N41 Q09 Q19 Q30 Q47 Q49; do
    echo "{\"index\": \"train-004-${file}-00000\", \"messages\": []}" >> train_indices.jsonl
done

# Dataset 5
echo '{"index": "train-005-new-00000", "messages": []}' >> train_indices.jsonl

# Dataset 6
echo '{"index": "train-006-new-00000", "messages": []}' >> train_indices.jsonl

echo "Creating eval indices (dataset 7)..."
> valid_indices.jsonl
echo '{"index": "eval-007-bootstrap-00000", "messages": []}' >> valid_indices.jsonl

echo "Done!"
wc -l train_indices.jsonl valid_indices.jsonl
