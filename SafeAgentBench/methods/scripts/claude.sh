cd ..



python -u generate_memory.py --model_name claude-sonnet-4-5-20250929 --use_bank true

counts=(0 20 40 60 80 100 120 140 160 180 200 220 240 260 269)
for count in "${counts[@]}"; do
    python -u pipeline_detailed.py --model_name claude-sonnet-4-5-20250929 --use_bank true --experiment true --bank_count $count 
done 

python -u pipeline_detailed.py --model_name claude-sonnet-4-5-20250929 --long_sys true --use_bank true --experiment true --bank_count 0 

python -u generate_memory.py --model_name claude-sonnet-4-5-20250929 --framework awm --workers 1
python extract_workflow.py --model_name claude-sonnet-4-5-20250929
python -u pipeline_detailed.py --model_name claude-sonnet-4-5-20250929 --use_bank false --workflow true
