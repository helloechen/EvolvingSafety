cd ..


# python -u generate_memory.py --model_name deepseek-v3.2 --use_bank true

# counts=(0 20 40 60 80 100 120 140 160 180 200 220 240 260 269)
counts=(269)
for count in "${counts[@]}"; do
    python -u pipeline_benign.py --model_name deepseek-v3.2 --use_bank true --experiment true --bank_count $count 
done 

# python -u pipeline_detailed.py --model_name deepseek-v3.2 --long_sys true --use_bank true --experiment true --bank_count 0 
# python -u pipeline_detailed.py --model_name gpt-4o-2024-11-20 --long_sys true --use_bank true --experiment true --bank_count 0 