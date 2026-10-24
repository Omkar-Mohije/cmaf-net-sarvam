# CMAF-Net | Sarvam Workshop App

**Two honest deliverables:**

- `app.py`: lightweight public Streamlit demo running real Sarvam Saaras v4 transcription, Sarvam-105B text sentiment and multilingual explanation; plus image/audio diagnostics, missing-modality simulator. Does **not** pretend Sarvam text sentiment is trained CMAF fusion.
- `research/`: PyTorch gated cross-modal attention training/inference scaffold from the earlier CMAF-Net build. Requires a real paired dataset and trained checkpoint. See `research/README.md`.

## Run locally

Use Python 3.11 or 3.12:

```bash
python -m pip install -r requirements.txt
# Set a NEW Sarvam API key: Windows PowerShell: $env:SARVAM_API_KEY="YOUR_NEW_KEY"
# macOS/Linux: export SARVAM_API_KEY="YOUR_NEW_KEY"
streamlit run app.py
```

Do not paste any secret into GitHub. **Revoke the API key shared in chat** and generate a new one.

## Publish to a URL using Streamlit Community Cloud

1. Upload contents of this folder to a new GitHub repository (upload `app.py`, `sarvam_client.py`, `signal_metrics.py`, `requirements.txt`, `.streamlit/config.toml`; optional `research/`).
2. Open https://share.streamlit.io and connect your GitHub account.
3. Create app → select repository and `main` branch → set entrypoint `app.py`.
4. Set app secrets to `SARVAM_API_KEY = "YOUR_NEW_KEY"`. This belongs in **Advanced settings / Secrets**, not a GitHub file.
5. Deploy. Share your assigned `https://<your-subdomain>.streamlit.app` URL.

If Cloud sleeps or rate limits, wake the app/retry; Sarvam usage may consume account credits. The UI requires audio ≤30 seconds for synchronous STT. Audio signal diagnostics are PCM WAV only; MP3 and M4A still work for transcription.

## Verification

```bash
python -m pytest -q tests
python -m compileall -q app.py sarvam_client.py signal_metrics.py
```

Tests mock network calls; live Sarvam API functionality needs an active key, credits and outbound internet. No performance metrics or trained checkpoints are bundled.

Docs: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/rest-api ; https://docs.sarvam.ai/api/api-guides-tutorials/chat-completion/overview
