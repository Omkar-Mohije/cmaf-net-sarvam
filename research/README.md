# CMAF-Net — Multimodal Resilience Lab

A Streamlit + PyTorch research prototype for **tri-modal sentiment classification** with gated cross-attention and modality-drop regularization. Includes optional Sarvam AI speech-to-text.

## Critical security note

**Never commit or paste API keys into the app.** Use a **newly rotated** Sarvam key after revoking any key shared in chat. The code contains no supplied credentials.

## Quick start

Python **3.11** is recommended. A GPU speeds up feature extraction, but CPU works for small inputs.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For optional Sarvam transcription, set `SARVAM_API_KEY` locally before launching the app (Windows PowerShell: `$env:SARVAM_API_KEY="NEW_KEY"`; macOS/Linux: `export SARVAM_API_KEY="NEW_KEY"`). Do not commit `.env` or `.streamlit/secrets.toml`.

**The app intentionally refuses to generate predictions until a real trained checkpoint exists.** The first run downloads pretrained ResNet-18 and DistilBERT weights from their model repositories.

## Data contract and training

Create a CSV at `data/manifest.csv` with columns `image,audio,text,label,split`. See `examples/manifest.example.csv`. Relative image/audio paths are resolved relative to the CSV directory. Leave paths/text blank when a modality is unavailable. Audio files for training must be WAV readable by `soundfile`. For the first iteration, use at least two sentiment classes and adequate independent examples per class in each split.

**Important:** CMU-MOSI/MOSEI datasets are not automatically bundled or downloaded. Their video/feature distribution, preprocessing and usage terms must be checked separately; the manifest loader expects your own paired media files and transcripts. You must build the manifest from properly aligned samples. Split by original video/speaker as appropriate **before** training to avoid leakage.

```bash
python train.py --manifest data/manifest.csv --epochs 8 --batch-size 8
streamlit run app.py
```

Training extracts frozen pretrained ResNet-18 and DistilBERT representations and log-Mel audio features once; the **trainable** module is CMAF-Net. It compares naive concat, ungated-dropout CMAF, modality-drop CMAF, and unimodal probes. Outputs in `artifacts/`: model checkpoints and `ablation.csv` with real held-out accuracy/macro-F1 under full input and leave-one-modality-out settings. Validation chooses best epoch, test is evaluated after selection.

For rigor, run multiple random seeds, confidence intervals, add noise sweeps across severity, and examine class imbalance before reporting results. This starter performs one fixed-seed run. Cross attention weights are not guaranteed explanations. Some implementation simplifications (frozen encoders, fixed log-Mel frames) should be clearly stated during viva.

## Troubleshooting

- First encoder download requires internet.
- 401/403 on Sarvam: check *new* API key, account permissions, credits.
- Sarvam synchronous transcription supports recordings up to ~30 seconds. Longer audio requires batch API.
- WAV upload is intentionally used for both local spectral processing and Sarvam speech transcription.
- No checkpoint? Supply a real paired dataset and train. Do not substitute random predictions for evaluation.

## Official documentation

- https://docs.sarvam.ai/api/getting-started/quickstart
- https://docs.sarvam.ai/api-reference/speech-to-text/transcribe
- https://docs.sarvam.ai/api/getting-started/models/saaras
