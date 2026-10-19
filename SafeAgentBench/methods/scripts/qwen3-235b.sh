cd ..


python -u generate_memory.py --model_name qwen3-235b-a22b-instruct-2507 --use_bank true

counts=(0 20 40 60 80 100 120 140 160 180 200 220 240 260 269)
for count in "${counts[@]}"; do
    python -u pipeline_detailed.py --model_name qwen3-235b-a22b-instruct-2507 --use_bank true --experiment true --bank_count $count 
done 

python -u pipeline_detailed.py --model_name qwen3-235b-a22b-instruct-2507 --long_sys True --use_bank true --experiment true --bank_count 0