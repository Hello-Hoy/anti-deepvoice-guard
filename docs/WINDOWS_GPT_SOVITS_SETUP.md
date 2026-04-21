# Windows GPT-SoVITS / OpenVoice Setup

This project uses two isolated environments:

- `.venv-cloning` in `anti-deepvoice-guard` for preprocessing, evaluation, OpenVoice orchestration, and reports.
- `.venv-gpt-sovits` inside the upstream `GPT-SoVITS` checkout for GPT-SoVITS training and inference.

## GPT-SoVITS v4

Upstream checkout lives at `D:\Claude_Projects\GPT-SoVITS` on this machine (not `C:\Projects\`). Pin to the release you validated:

```powershell
Set-Location "D:\Claude_Projects\GPT-SoVITS"
git fetch --tags
git checkout 20250422v4
git log -1 --oneline
```

Record the SHA in the experiment report.

### Pretrained v4 layout (verified against HF `lj1995/GPT-SoVITS` on 2026-04-22)

Files live under:

```text
D:\Claude_Projects\GPT-SoVITS\GPT_SoVITS\pretrained_models\
```

Only two files are v4-specific; the rest are inherited from v2ProPlus or shared:

| Role | Relative path | Override env var | Source |
|---|---|---|---|
| S2 generator (v4) | `gsv-v4-pretrained/s2Gv4.pth` | `GPT_SOVITS_PRETRAINED_S2G` | HF `lj1995/GPT-SoVITS` |
| S2 discriminator | `v2Pro/s2Dv2ProPlus.pth` *(inherited — v4 ships no separate s2D)* | `GPT_SOVITS_PRETRAINED_S2D` | HF `lj1995/GPT-SoVITS` |
| S2 config template | `configs/s2.json` *(shared; `model.version` set programmatically)* | `GPT_SOVITS_S2_CONFIG_TEMPLATE` | repo |
| v4 vocoder | `gsv-v4-pretrained/vocoder.pth` | `GPT_SOVITS_VOCODER` | HF `lj1995/GPT-SoVITS` |
| BigVGAN 24k | `models--nvidia--bigvgan_v2_24khz_100band_256x` | `GPT_SOVITS_BIGVGAN_DIR` | HF `nvidia/bigvgan_v2_24khz_100band_256x` |
| S1 base | `s1v3.ckpt` *(shared with v3)* | `GPT_SOVITS_PRETRAINED_S1` | HF `lj1995/GPT-SoVITS` |
| CnHuBERT | `chinese-hubert-base/` | `GPT_SOVITS_CNHUBERT_DIR` | HF |
| BERT | `chinese-roberta-wwm-ext-large/` | `GPT_SOVITS_BERT_DIR` | HF |
| SV (speaker verification) | `sv/pretrained_eres2netv2w24s4ep4.ckpt` | `GPT_SOVITS_SV_CKPT` | HF |

All of the above are already present on this machine under the `D:\` path.

### CUDA Alignment (verified 2026-04-22)

Current `.venv-gpt-sovits` runs `torch 2.5.1+cu121`. Despite upstream release notes sometimes referencing CU126/CU128, `torch.cuda.is_available()=True` and CUDA 12.1 initializes cleanly on RTX 4080 Super. Do **not** upgrade torch preemptively. Only rebuild `.venv-gpt-sovits` if a v4 inference/training run raises a concrete CUDA-kernel error.

### Environment Variables

```powershell
$env:GPT_SOVITS_DIR = "D:\Claude_Projects\GPT-SoVITS"
$env:GPT_SOVITS_VENV = "D:\Claude_Projects\GPT-SoVITS\.venv-gpt-sovits\Scripts\python.exe"
$env:GPT_SOVITS_VERSION = "v4"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

### Health check

```powershell
.\.venv-cloning\Scripts\python.exe model-training\voice-clone\clone_my_voice.py --engine gpt_sovits --health-check
.\.venv-cloning\Scripts\python.exe model-training\voice-clone\scripts\phase_b_retrain.py --speaker hyohee --version v4 --dry-run
```

## OpenVoice v2 + MeloTTS (Korean zero-shot secondary track)

### Pinned upstream versions

- OpenVoice — no tagged releases; pinned to commit `74a1d147b17a` (2025-04-19).
- MeloTTS — pinned to release tag `v0.1.2` (2024-03-01).

Both pins are recorded in `model-training/requirements-cloning.txt`.

### Checkpoint paths (defaults in `engines/openvoice2.py`)

```powershell
$env:MELOTTS_KR_CKPT         = "$env:USERPROFILE\tools\MeloTTS\checkpoints\korean"
$env:OPENVOICE2_TONE_CONVERTER = "$env:USERPROFILE\tools\OpenVoice\checkpoints_v2\converter"
$env:OPENVOICE2_SE_EXTRACTOR   = "$env:USERPROFILE\tools\OpenVoice\checkpoints_v2\base_speakers\ses"
```

Adapt via env vars if you place the checkpoints elsewhere.

### Install and health check

```powershell
# After checkpoints are downloaded (see downloads section below):
.\.venv-cloning\Scripts\python.exe -m pip install `
  "git+https://github.com/myshell-ai/OpenVoice.git@74a1d147b17a#egg=openvoice" `
  "git+https://github.com/myshell-ai/MeloTTS.git@v0.1.2#egg=melotts"

.\.venv-cloning\Scripts\python.exe model-training\voice-clone\clone_my_voice.py --engine openvoice2 --health-check
```

### Downloads (OpenVoice v2 checkpoints_v2 + MeloTTS-KR)

OpenVoice v2 bundle: `https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip`
MeloTTS Korean checkpoint: downloaded automatically on first use via HuggingFace when the package is installed; cache lives under `%USERPROFILE%\.cache\huggingface\hub\`.

## Notes

- Inference upstream signatures may drift; if a future OpenVoice commit breaks the adapter, re-run with the pinned commit.
- Re-record `.venv-cloning` requirements with `pip freeze > .pip_after_setup.txt` after install, and commit the lockfile if you intend to reproduce later.
