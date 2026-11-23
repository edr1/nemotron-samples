# Data Science ML Agent

This example demonstrates how to build a <b>natural language-driven data science and machine learning agent</b> powered by <b>NVIDIA GPUs</b>.
The agent allows users to perform data exploration, model training, and hyperparameter optimization interactively using <b>RAPIDS cuDF</b> and <b>cuML</b> for GPU acceleration. 

## Overview

The Data Science ML Agent enables natural language interaction for common data science workflows, including dataset loading, target selection, model training, and performance optimization.
It combines the flexibility of LLMs with GPU-accelerated computation to simplify and speed up end-to-end machine learning pipelines.

## LLM Used

The default model is NVIDIA Nemotron 3.5 Lightning. Open **Ollama settings** in the Streamlit sidebar to enable Ollama and edit the server URL or model for the current session. `OLLAMA_BASE_URL` and `OLLAMA_MODEL` set the initial values.

## Use Ollama on a DGX Spark

On the Spark, with the Ollama service running, pull the model:

```bash
ollama pull nemotron-3.5-lightning:30b-a3b
```

In a PowerShell window on the notebook, open an SSH tunnel using your Spark username and hostname or IP. Keep this window open:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:11435:127.0.0.1:11434 USER@SPARK_HOST
```

If Ollama is already listening on a private network interface, you can set `OLLAMA_BASE_URL` to `http://SPARK_IP:11434` directly. Use the server root URL. The client adds `/api/chat`. The default Ollama tag is [`nemotron-3.5-lightning:30b-a3b`](https://ollama.com/library/nemotron-3.5-lightning%3A30b-a3b). Set `OLLAMA_MODEL` only if you want to use a different installed tag. Ollama handles chat and tool selection on the Spark. The notebook runs the selected Python tools and trains the ML models.

## Key Features

- Natural language interface for running data exploration and ML workflows
- GPU acceleration with cuDF (DataFrame operations) and cuML (ML algorithms)
- Support for CPU mode using pandas and scikit-learn
- Simple setup and execution through Streamlit interface
- Compatible with both small and large-scale datasets

## Run App with Python and pip

The CPU mode uses pandas and scikit-learn. With Ollama enabled, the model selects a tool. With Ollama disabled, the chat recognizes common phrases such as `load dataset train.csv` and lets you call every registered tool explicitly, for example `set_target("Survived")`, `preview_data(5)`, or `optimize_svc(50)`. Enter `help()` for the tool list. Both modes execute the same Python tools.

From this example's directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$env:OLLAMA_BASE_URL = "http://10.8.0.32:11434"
(Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags").models | Select-Object name
.\.venv\Scripts\python.exe -m streamlit run user_interface.py --server.address 127.0.0.1
```

From this example's directory in Linux:
```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

export OLLAMA_BASE_URL="http://127.0.0.1:11434"
curl -fsS "${OLLAMA_BASE_URL}/api/tags" | jq -r '.models[].name'
./.venv/bin/python -m streamlit run user_interface.py --server.address=10.8.0.32 --server.port=8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). Upload `train.csv` and enter `Load train.csv with Survived as the target`, then `Train classification models`. Upload `test.csv` and enter `Make predictions for test.csv` to try inference.

## Automatic ML backend on DGX Spark

- RAPIDS 25.10
- Python 3.10, 3.11, 3.12, or 3.13
- CUDA 12.0 or 13.0 compatible NVIDIA GPU (for GPU mode)
- DGX Spark or an equivalent GB10 system running Linux on ARM64

Please refer to the [official RAPIDS installation documentation](https://docs.rapids.ai/install/) for detailed instructions.

Create a separate environment from the example's directory using a compatible Python interpreter. For example, with Python 3.12 and the CUDA 12 Toolkit:
```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com \
    'cudf-cu12==25.10.*' 'cuml-cu12==25.10.*'
python -m streamlit run user_interface.py --server.address 127.0.0.1
```

The app detects a local GB10 with `nvidia-smi` and enables `cudf.pandas` and `cuml.accel` before importing pandas and scikit-learn. The sidebar shows the selected backend. On other machines, or when RAPIDS packages or a CUDA device are unavailable, the pipeline runs on CPU. On RAPIDS, operations without GPU support can still run on CPU. The backend is selected for the Python process running Streamlit, independently of the Ollama server.

## Example Usage

Example queries:
```bash
load dataset train.csv 
set target variable to be 'Survived'
train classification/regression model
optimize svc with 50 trials
optimize forest regressor with 30 trials
show best model by r2
make inference for the test dataset
...
```

## 📊 Sample Dataset

This project provides sample datasets, you can download the Kaggle Titanic Datasets [train.csv](https://www.kaggle.com/competitions/titanic) and [test.csv](https://www.kaggle.com/competitions/titanic) into the data folder. You can also create **train-1M**, an extrapolated version scaled to 1M rows of the train.csv using the script extrapolation.py. 


**Note:**  
- Ensure you have the appropriate dependencies installed for each mode.  
- GPU mode requires a supported NVIDIA GPU and the RAPIDS ecosystem installed.
