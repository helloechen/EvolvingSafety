
cd ..


# python -u pipeline_detailed.py --model_name Qwen3-8B --long_sys True --use_bank true --experiment true --bank_count 0
counts=(0 20 40 60 80)
for count in "${counts[@]}"; do
    python -u pipeline_detailed.py --model_name Qwen3-8B --use_bank true --experiment true --bank_count $count 
done 