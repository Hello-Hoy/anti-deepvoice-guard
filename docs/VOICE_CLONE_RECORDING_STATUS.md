# Voice-clone 녹음 진행 상황 (Phase A)

`voice-clone/v4-hybrid-setup` 브랜치의 Mac 측 작업 결과. RTX 4080 데스크탑에서 `git pull` 후 이 문서를 기준으로 다음 단계(Phase B 학습)를 이어 받습니다.

## 화자
- speaker: `hyohee`
- 녹음 일시: 2026-04-22

## 결과 요약

| 지표 | 값 |
|---|---|
| Phase A 녹음 | 120/120 clip, 모두 accepted |
| Phase A SNR | 평균 34.6dB / 최저 29.2 / 최고 37.8 |
| Phase A peak | 평균 -11.1dBFS |
| 전처리 채택 | 84/120 segments |
| 전처리 총 길이 | 348초 (5.8분) |
| split | train 63 / eval 8 / test 13 (75/10/15, seed=42) |
| Phase 0 ref | `phase0_ref_20260422_clip014.wav` (Phase A clip_014 재사용, SNR 37.4 / 6.2초) |

## 외장 디스크 (X31) 레이아웃

```
/voice-clone-data/
├── raw_hyohee/
│   └── session_20260422_220351/         # 원본 48kHz/24bit WAV 120개 + session_log.json
├── user-hyohee_preprocessed/
│   ├── manifest.json                    # GPT-SoVITS prepare_data 입력
│   ├── segment_index.json
│   ├── wavs_24k/                        # 학습용
│   ├── wavs_16k/                        # 평가용
│   └── _workspace/
├── phase0_ref_20260422_clip014.wav      # Phase 0 합성 ref
└── voice-clone-data_20260422.tar.gz     # 위 3개 압축본 (백업)
```

## Windows 측 작업 순서

1. `git pull` (이 브랜치 최신 commit 받기)
2. 외장 디스크 → `model-training/voice-clone/data/`로 복사
   - `raw_hyohee/session_*` → `data/raw/hyohee/`
   - `user-hyohee_preprocessed/*` → `data/user-hyohee/`
   - `phase0_ref_*.wav` → `test-samples/voice-clone-raw/`
3. `docs/WINDOWS_GPT_SOVITS_SETUP.md` 4단계(venv) → 7단계(평가)까지 진행
4. Phase B 학습 시 `manifest_path` = `model-training/voice-clone/data/user-hyohee/manifest.json`

## record_helper 패치 (이 브랜치에 포함)

녹음 중 발견한 두 가지를 같이 commit:
1. interactive 모드의 1시간 미리할당 버퍼 버그 → callback 기반 `InputStream`으로 교체. `sd.stop()` 후 실제 녹음 길이만 반환됨.
2. `MIN_SNR_DB` 35 → 25. 코드의 SNR 정의(`full-clip RMS / room tone RMS`)는 발화 비율에 매우 민감해서, 5~10초 자연 발화 + 끝 여백 조합에서도 28~37dB 범위. 35dB 컷오프는 이 측정 방식엔 비현실적이라 25로 완화. 전처리(`preprocess_recordings.py --min-snr-db`)도 동일 25 기준 사용.

## 알려진 한계

- 5.8분 발화는 GPT-SoVITS v4 fine-tune의 최저선. 학습 후 화자 유사도/한국어 발음/일관성 평가에서 부족하면 추가 녹음 필요.
- 전처리 단계에서 36/120(30%)가 컷됨. 기록만 보면 `record_helper`는 noise floor를 room tone에서 한 번만 잡지만 `preprocess_recordings.py`는 segment별 재측정해서 더 엄격하게 골라냄. 다음 라운드 녹음 시 이 격차를 고려해 record_helper threshold를 30 정도로 올려도 좋음.
