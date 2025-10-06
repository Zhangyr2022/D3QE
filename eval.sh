sub_dirs=("Infinity" "Janus_Pro" "RAR" "Switti" "VAR" "LlamaGen" "Open_MAGVIT2")

python eval.py \
    --model_path /path/to/your/model.pth \
    --detect_method D3QE  \
    --batch_size 1 \
    --dataroot /path/to/your/testset \
    --sub_dir "${sub_dirs[@]}"  