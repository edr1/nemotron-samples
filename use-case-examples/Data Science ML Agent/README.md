# Data Science ML Agent

This example demonstrates how to build a <b>natural language-driven data science and machine learning agent</b> powered by <b>NVIDIA GPUs</b>.
The agent allows users to perform data exploration, model training, and hyperparameter optimization interactively using <b>RAPIDS cuDF</b> and <b>cuML</b> for GPU acceleration. 

## Overview

The Data Science ML Agent enables natural language interaction for common data science workflows, including dataset loading, target selection, model training, and performance optimization.
It combines the flexibility of LLMs with GPU-accelerated computation to simplify and speed up end-to-end machine learning pipelines.

## LLM Used

The default model is NVIDIA Nemotron 3.5 Lightning. Set `OLLAMA_BASE_URL` to run inference through Ollama.

## Use Ollama on a DGX Spark

On the Spark, with the Ollama service running, pull the model:

```bash
ollama pull nemotron-3.5-lightning:30b-a3b
```

In a PowerShell window on the notebook, open an SSH tunnel using your Spark username and hostname or IP. Keep this window open:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:11435:127.0.0.1:11434 USER@SPARK_HOST
```

In another PowerShell window, from this example's directory, verify the tunnel and start Streamlit:

```powershell
$env:OLLAMA_BASE_URL = "http://10.8.0.32:11434"
(Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags").models | Select-Object name
.\.venv\Scripts\python.exe -m streamlit run user_interface.py --server.address 127.0.0.1
```

If Ollama is already listening on a private network interface, you can set `OLLAMA_BASE_URL` to `http://SPARK_IP:11434` directly. Use the server root URL. The client adds `/api/chat`. The default Ollama tag is [`nemotron-3.5-lightning:30b-a3b`](https://ollama.com/library/nemotron-3.5-lightning%3A30b-a3b). Set `OLLAMA_MODEL` only if you want to use a different installed tag. Ollama handles chat and tool selection on the Spark. The notebook runs the selected Python tools and trains the ML models.

## Key Features

- Natural language interface for running data exploration and ML workflows
- GPU acceleration with cuDF (DataFrame operations) and cuML (ML algorithms)
- Support for CPU mode using pandas and scikit-learn
- Simple setup and execution through Streamlit interface
- Compatible with both small and large-scale datasets

## Run on Windows with Python and pip

The CPU mode uses pandas and scikit-learn. When Ollama is configured, the chat sends the commands below to that model for tool selection. When `OLLAMA_BASE_URL` is unset, these commands run directly in Python.

From this example's directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run user_interface.py --server.address 127.0.0.1
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). Upload `train.csv` and enter `Load train.csv with Survived as the target`, then `Train classification models`. Upload `test.csv` and enter `Make predictions for test.csv` to try inference.

## GPU mode with Python and pip

- RAPIDS 25.10
- Python 3.10, 3.11, 3.12, or 3.13
- CUDA 12.0 or 13.0 compatible NVIDIA GPU (for GPU mode)
- Linux or Ubuntu on WSL2 for Windows

Please refer to the [official RAPIDS installation documentation](https://docs.rapids.ai/install/) for detailed instructions.

Create a separate environment from the example's directory using a compatible Python interpreter. For example, with Python 3.12 and the CUDA 12 Toolkit:
```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com \
    'cudf-cu12==25.10.*' 'cuml-cu12==25.10.*'
python -m cudf.pandas -m cuml.accel -m streamlit run user_interface.py
```

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
