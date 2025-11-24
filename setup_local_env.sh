#!/bin/bash

# ============ 1. Basics ============
echo ">>> Setting up local environment for Cross-Modal TinyLlama project..."

if ! command -v conda &> /dev/null
then
    echo "conda not found. Please install Miniconda or Anaconda first."
    exit
fi

# ============ 2. Create Conda Environment ============
ENV_NAME="10701"
PYTHON_VER="3.11"

echo ">>> Creating conda environment: $ENV_NAME"
mamba create -y -n $ENV_NAME python=$PYTHON_VER
eval "$(mamba shell hook --shell bash)"
mamba activate $ENV_NAME

# ============ 3. Install PyTorch (CPU or GPU version) ============

mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
# echo ">>> Installing PyTorch (CPU version)..."
# conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

# ============ 4. Install requirements ============
echo ">>> Installing project dependencies..."
# pip install transformers==4.45.1
# pip install datasets==2.20.0
# pip install numpy==1.26.4
# pip install scikit-learn==1.5.0
# pip install matplotlib==3.8.4
# pip install mlflow==2.14.3
# pip install tqdm==4.66.2
mamba install -y transformers datasets numpy scikit-learn matplotlib mlflow tqdm -c conda-forge

# ============ 5. Initialize MLflow ============
echo ">>> Setting up local MLflow tracking directory..."
mkdir -p ./mlruns
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
echo "MLflow tracking URI set to local folder: $(pwd)/mlruns"

# ============ 6. Start MLflow UI（可选） ============
echo ">>> You can start MLflow UI anytime by running:"
echo "mlflow ui --port 5000"
echo "Then open http://127.0.0.1:5000 in your browser."

# ============ 7. Finish ============
echo "✅ Environment setup complete! Now run:"
echo "mamba activate $ENV_NAME"
echo "python train_baseline.py"
