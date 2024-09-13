#!/bin/bash

sorted_files=$(ls *.csv | sort -V)

for file in $sorted_files; do
    python3 calculate.py $file
done