export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"

# Note: To infer Chinese dialects, set model_dir to "pretrained_models/SoulX-Podcast-1.7B-dialect
model_dir=pretrained_models/SoulX-Podcast-1.7B
input_file=example/podcast_script/script_mandarin.json

python cli/podcast.py \
        --json_path ${input_file} \
        --model_path ${model_dir} \
        --output_path outputs/mandarin.wav \
        --seed 7