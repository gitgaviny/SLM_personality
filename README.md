# SLM Personality

Code for "Personality-Aware Spoken Language Modeling with Token-Partitioned Multitask Learning for Speech Emotion Recognition".

## Quick Start

Create an environment and install the main dependencies:

```bash
python -m venv path/to/your/venv
source path/to/your/venv/bin/activate
pip install torch transformers datasets peft accelerate safetensors pandas soundfile librosa evaluate
```

Install a CUDA-compatible PyTorch build if needed.

## Data

The CSV files in `iemocap/` use placeholder wav paths such as:

```text
path/to/your/wav/Ses01F_impro01_F000.wav
```

Replace `path/to/your/wav` with your local IEMOCAP wav directory. Then generate the ESPnet-style files:

```bash
cd iemocap_special_token_serasr
python process.py
```

If needed, edit `CSV_DIR` in `iemocap_special_token_serasr/process.py`.

## Pipeline

The main entrypoint is `src/iemocap.sh`.

- `stage=1`: build Hugging Face datasets from ESPnet-style files
- `stage=2`: create the speech encoder + Llama decoder model
- `stage=3`: fine-tune the model
- `stage=4`: run inference and compute WER / accuracy

Common placeholders to replace:

- `path/to/your/venv`: Python virtual environment
- `path/to/data`: directory containing `sessionX/train` and `sessionX/test`
- `path/to/model`: parent directory containing the Llama checkpoint
- `/path/to/stage1/sessionX`: pretrained speech encoder checkpoint for each session
- `CHANGEME_PARTITION`: your Slurm partition

## Run with Slurm

Edit `src/slurm/preprocess.slurm` and `src/slurm/train_infer.slurm` for your cluster, then run:

```bash
cd src/slurm
sbatch preprocess.slurm
sbatch train_infer.slurm
```
