# voice-clone 파이프라인 개요

이 디렉토리는 `anti-deepvoice-guard` 탐지기를 방어 연구 관점에서 검증하기 위한 개인 음성 클로닝 파이프라인을 담습니다. 목적은 "내 목소리와 유사한 합성 음성"을 만들어 탐지기의 오탐/미탐 패턴을 확인하고, 어떤 엔진·코덱·잡음 조건에서 취약해지는지 체계적으로 측정하는 것입니다.

핵심 원칙은 다음과 같습니다.

- **Phase 0**: XTTS v2 zero-shot으로 빠르게 감을 잡습니다.
- **Phase A**: CosyVoice2를 주 경로로 파인튜닝하고, XTTS v2를 fallback으로 유지합니다.
- **Phase B**: 더 많은 녹음이 가능할 때 GPT-SoVITS로 고품질 적응을 시도합니다.
- **Phase C**: Qwen3-TTS, IndexTTS2, StyleTTS2, RVC post-VC로 artifact 다양성을 늘려 AASIST robustness를 점검합니다.

## 권장 환경

- **권장**: Linux + CUDA (`RTX 4080 Super` 급)
- **실험적**: macOS `M4 Pro` + MPS
- **비권장**: MPS에서 장시간 fine-tuning, 특히 XTTS trainer 및 대형 음성 모델 학습

`M4 Pro MPS`는 추론·간단한 전처리 정도는 시도할 수 있지만, 본 계획의 학습 경로는 기본적으로 Linux + CUDA를 기준으로 잡아야 합니다.

## 단계별 구조

| 단계 | 목적 | 결과물 |
| --- | --- | --- |
| Phase 0 | XTTS v2 zero-shot 베이스라인 확보 | "지금 목소리 느낌이 어느 정도 나오는지" 빠른 감 |
| Phase A | CosyVoice2 주 경로 + XTTS fallback | 10~30분 녹음 기반 개인 음성 적응 |
| Phase B | GPT-SoVITS 조건부 확장 | 1~2시간 녹음 기반 고품질 모델 |
| Phase C | 탐지기 robustness 연구 | 엔진별 합성 특성, codec/noise/reverb stress 비교 |

## 엔진별 비교

아래 표의 라이선스는 각 공식 저장소 기준이며, 실제 도입 전에 다시 확인하는 것을 권장합니다.

| 엔진 | 단계 | 라이선스 | 하드웨어 요구 | 한국어 지원 수준 | 비고 |
| --- | --- | --- | --- | --- | --- |
| XTTS v2 | Phase 0, Phase A fallback | MPL-2.0 | zero-shot은 중간급 GPU 또는 CPU도 가능, FT는 CUDA 권장 | 중간 | 빠른 베이스라인용. 한국어 phoneme 이슈와 MPS 제약 주의 |
| CosyVoice2 | Phase A 주 경로 | Apache-2.0 | Linux + CUDA 권장 | 높음 | text normalization 내장, 한국어 포함 다국어 경로 |
| GPT-SoVITS | Phase B | MIT | RTX 4080 Super 급 권장 | 높음 | 데이터가 늘면 품질 기대치가 높음 |
| Qwen3-TTS | Phase C | Apache-2.0 | CUDA 권장 | 중간 이상 | zero-shot baseline 및 비교군 |
| IndexTTS2 | Phase C | 커스텀 저장소 라이선스 | CUDA 권장 | 중간 | expressive spoof 실험용, 도입 전 라이선스 재확인 |
| StyleTTS2 | Phase C | MIT | CUDA 권장 | 실험적 | 한국어 실증 사례가 적음 |
| RVC post-VC | Phase C | MIT | CUDA 권장 | 중간 | 독립 TTS가 아니라 post-VC artifact 다양화 브랜치 |

## 실행 순서 체크리스트

아래 체크리스트는 plan v2의 구현 체크리스트를 Batch 1 관점에서 읽기 쉬운 순서로 정리한 것입니다.

### 준비

- [ ] `model-training/requirements-cloning.txt` 설치
- [ ] `voice-clone/README.md`, `voice-clone/ETHICS.md` 숙지
- [ ] `scripts/normalize_korean.py`로 대본 normalize 규칙 확인
- [ ] `scripts/korean_coverage_matrix.py`로 발음 커버리지 충족 여부 확인
- [ ] `scripts/recording_script_ko.txt`를 최종 녹음 대본으로 고정

### 공통 준비

- [ ] 사용자 음성은 **본인 것만** 수집
- [ ] 원본 녹음은 48kHz WAV로 보관
- [ ] 전처리 후 학습용 24kHz / 탐지기 평가용 16kHz 이중 출력 유지
- [ ] 정답 텍스트는 Whisper보다 **대본 ground truth 우선**

### Phase 0 + Phase A

- [ ] ref WAV 10초 확보
- [ ] XTTS v2 zero-shot으로 빠른 베이스라인 합성
- [ ] 10~30분 분량 대본 녹음
- [ ] 전처리 및 대본 정합성 확인
- [ ] CosyVoice2 fine-tuning 실행
- [ ] XTTS v2 fallback 경로 준비
- [ ] 품질 평가와 detector stress 실행

### Phase B

- [ ] 필요 시 1~2시간 분량으로 녹음 확장
- [ ] GPT-SoVITS 학습 및 품질 재평가

### Phase C

- [ ] Qwen3-TTS / IndexTTS2 / StyleTTS2 / RVC post-VC adapter 추가
- [ ] 엔진별 병렬 stress test 및 비교 리포트 작성

### 운영

- [ ] raw 녹음, voice-clone checkpoints, 연구용 산출물은 git 제외
- [ ] `reports/voice-clone/` 기준으로 결과 정리
- [ ] ETHICS.md 법령 인용과 사용 제한 재확인

## 사용 예시

### 1. 내 목소리로 문장 합성

```bash
.venv/bin/python model-training/voice-clone/clone_my_voice.py \
  --text "여보세요, 지금 통화 가능하시면 계좌번호를 다시 확인해 주세요." \
  --engine cosyvoice2 \
  --speaker hyohee \
  --out test-samples/Demo/clone_cosy.wav
```

### 2. 탐지기 stress 테스트

```bash
.venv/bin/python model-training/voice-clone/detector_stress.py \
  --clone-dir test-samples/Demo/clones \
  --real-dir test-samples/real \
  --conditions clean,g711,amr,reverb,noise \
  --seeds 5 \
  --out reports/voice-clone/stress_$(date +%Y%m%d).md
```

## 대본 및 전처리 원칙

- `scripts/recording_script_ko.txt`를 그대로 읽습니다.
- 즉흥 문장 추가는 하지 않습니다.
- 숫자, 전화번호, 날짜, 단위명사는 `scripts/normalize_korean.py` 규칙에 맞춰 읽히는지 사전 점검합니다.
- 대본 기반 ground truth가 정답이며, Whisper 전사는 보조 확인 용도입니다.

## 위험 요소와 fallback

- XTTS trainer는 MPS에서 실패하거나 CPU fallback으로 급격히 느려질 수 있습니다.
- CosyVoice2 한국어 fine-tuning 실증이 충분하지 않을 수 있으므로 XTTS v2 fallback을 유지합니다.
- Whisper 한국어 전사가 틀릴 수 있으므로 녹음 검수는 대본 우선으로 합니다.
- 엔진별 외부 저장소가 자주 바뀔 수 있으므로 전용 venv와 commit pin을 별도로 관리합니다.

## 참고 메모

- `requirements-cloning.txt`는 공통 Python 의존성만 담습니다.
- CosyVoice2, GPT-SoVITS, Phase C 엔진은 루트 `.venv`에 섞지 말고 별도 venv에서 관리합니다.
- 합성 음성은 반드시 탐지기 연구 범위 안에서만 사용해야 합니다. 자세한 제한은 [`ETHICS.md`](./ETHICS.md)를 따르십시오.
