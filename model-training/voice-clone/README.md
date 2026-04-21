# Voice Clone Pipeline

This directory contains the research-only voice-clone pipeline used to create evaluation assets for Anti-DeepVoice Guard.

## Hybrid Plan

- Primary: GPT-SoVITS v4 fine-tune, explicit `--version v4`, 48 kHz output path, no silent fallback to older versions.
- Secondary: OpenVoice v2 + MeloTTS-KR zero-shot baseline for comparison and future multi-speaker expansion.
- Both engines report two reference tracks: `rotate` and `best`.

`rotate` uses round-robin references selected from train-split SNR top clips, 6-10 seconds long. `best` chooses a reference by simple text-length/style/SNR heuristics. Reports should include both tracks.

## Recording Script

`scripts/recording_script_ko.txt` contains 120 Korean sentences. It was expanded from the original short script to cover news, dialogue, joy, sadness, calm, firm anti-fraud prompts, short utterances, and long-form utterances. The file includes a provenance comment at the top.

## Data Gates

Preprocessing defaults to:

- `--min-snr-db 35`
- clipping rejection
- true peak rejection when `> -1 dBFS`
- split ratio `75/10/15`
- split seed `42`

Split assignment and rejection metadata are written into the manifest.

## Evaluation

The objective evaluation stack is five-stage:

1. WER using Whisper large-v3, hard threshold `< 0.10`.
2. ECAPA-TDNN cosine using SpeechBrain `spkrec-ecapa-voxceleb`.
3. DNSMOS P.835 OVRL.
4. UTMOS via `UTMOS_PREDICT_CMD`.
5. AASIST fake score as detector-facing information.

`eval_quality.py` writes JSON and markdown reports with `gpt_sovits` and `openvoice2` sections split by `rotate` and `best`.

## Human MOS

`scripts/human_mos_tool.py` serves a randomized 30-clip pool:

- real 10
- GPT-SoVITS best-ref 10
- OpenVoice v2 best-ref 10

Each clip asks for real-vs-AI, naturalness 1-5, and speaker similarity 1-5, with max two plays. It writes per-listener CSV and aggregate JSON with deception rate and Wilson 95% lower bound.

## Detector Stress

`detector_stress.py` supports clean, G.711, AMR-NB, Opus, MP3 128/64, AAC 96, telephony bandpass, AGC, compressor, soft clipping, packet loss 1/3/5, resample chain 16-8-16, noise, and reverb conditions.

## Rollback

If GPT-SoVITS v4 fails because of Windows/CUDA mismatch, keep the new data gates, 120-sentence script, conservative sampling, and dual ref-mode evaluation, then rerun with the last known stable v2ProPlus version explicitly. Do not rely on implicit version fallback.
