# Running `run_subtask1_baselines_rocm.sh`

This README explains how to set up the environment and run the ROCm/AMD baseline script:

```bash
run_subtask1_baselines_rocm.sh
```

## Requirements

Install **Python 3.12** or **Python 3.11** before continuing.

This setup is intended for machines with **AMD GPUs using ROCm**.

## 1. Create and activate a virtual environment

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

You should now see `(.venv)` in your shell prompt.

## 2. Install AMD/ROCm baseline dependencies

Install the ROCm-compatible requirements:

```bash
pip install -r baselines/requirements-amd.txt
```

## 3. Clone the SemEval 2026 Task 11 repository

Clone the official task repository:

```bash
git clone https://github.com/neuro-symbolic-ai/semeval_2026_task_11.git
```

## 4. Add the Qwen3-8B model

The baseline expects access to the **Qwen3-8B** model.

### Option A: Copy the model from Fox

If you have access to Fox, copy the model with:

```bash
mkdir -p ./hf_models

rsync -avP YOURUSERNAME@fox.educloud.no:/fp/projects01/ec403/hf_models/Qwen3-8B ./hf_models/
```

Replace `YOURUSERNAME` with your Fox username.

After this, the model should be available at:

```text
./hf_models/Qwen3-8B
```

### Option B: Download the model from Hugging Face

The model on Fox may be the same as the Hugging Face version. If you do not have access to Fox, you can instead download the Qwen3-8B model from Hugging Face and place it under:

```text
./hf_models/Qwen3-8B
```

For example, using the Hugging Face CLI:

```bash
pip install huggingface_hub
mkdir -p ./hf_models

huggingface-cli download Qwen/Qwen3-8B \
  --local-dir ./hf_models/Qwen3-8B
```

You may need to log in first if the model requires authentication:

```bash
huggingface-cli login
```

## 5. Run the ROCm baseline script

Once the environment, dependencies, repository, and model are ready, run:

```bash
bash run_subtask1_baselines_rocm.sh
```

## Notes
* Ensure the model path used by the script matches:

```text
./hf_models/Qwen3-8B
```

