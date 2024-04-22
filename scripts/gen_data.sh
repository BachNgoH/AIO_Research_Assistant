CUDA_VISIBLE_DEVICES=2,3 python3 generate_data.py \
    --dataset_name="BachNgoH/ParsedArxivPapers_12k" \
    --load_local=True \
    --service="vllm" \
    --model_name="BachNgoH/Gemma_7b_Citation_16bit" \
    --output_path="./outputs" \
