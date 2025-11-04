FROM vllm/vllm-openai:v0.10.1
RUN git clone https://github.com/Soul-AILab/vllm.git && \
	cd vllm && \
        git checkout v0.10.1.1-soulxpodcast

# Get the default VLLM path and replace the default file to edited one.
RUN VLLM_PATH=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null) && \
    echo "Docker VLLM Path: $VLLM_PATH" && \
    cp "./vllm/vllm/model_executor/layers/sampler.py" "${VLLM_PATH}/model_executor/layers/sampler.py" && \
    cp "./vllm/vllm/model_executor/layers/utils.py" "${VLLM_PATH}/model_executor/layers/utils.py" && \
    cp "./vllm/vllm/model_executor/sampling_metadata.py" "${VLLM_PATH}/model_executor/sampling_metadata.py" && \
    cp "./vllm/vllm/sampling_params.py" "${VLLM_PATH}/sampling_params.py"

# Install the SoulX-Podcast requirement
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --no-cache-dir \
	librosa \
	numpy \
	scipy \
	s3tokenizer \
	diffusers \
	torch==2.7.1 \
	torchaudio==2.7.1 \
	transformers==4.57.1 \
	accelerate==1.10.1 \
	onnxruntime \
	onnxruntime-gpu \
	einops \
	gradio \
	triton>=3.0.0 \
    huggingface-hub==0.34.0

# Reset VLLM default api server entrypoint
ENTRYPOINT ["/bin/bash"]