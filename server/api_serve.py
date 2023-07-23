!python -m vllm.entrypoints.openai.api_server \
    --port $CDSW_APP_PORT \
    --host 127.0.0.1 \
    --model ./models/vicuna-7b-v1.3 \
    --tokenizer ./models/llama-tokenizer