#!/bin/bash

for val in  nicolinho/QRM-Llama3-8B maywell/Better-PairRM openbmb/Eurus-RM-7b; do
    python new_scores_existing_results.py --r_name $val
done

for val in 0 1 2 5 6 7 8 9 10 11 12 13 14 15; do
    python speed_test.py --r_idx $val
done