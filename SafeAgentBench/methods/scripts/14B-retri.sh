

cd ..

counts=(20 40 60 80 100 120 140 160 180 200 220 240 260 269)
for count in "${counts[@]}"; do
    python -u pipeline_detailed.py --model_name Qwen3-14B --retrival_num 5 --use_bank true --experiment true --bank_count $count 
done 

counts=(20 40 60 80 100 120 140 160 180 200 220 240 260 269)
for count in "${counts[@]}"; do
    python -u pipeline_detailed.py --model_name Qwen3-14B --retrival_num 7 --use_bank true --experiment true --bank_count $count 
done 
