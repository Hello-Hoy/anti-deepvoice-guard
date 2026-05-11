# 2026-05-11 — GPT-SoVITS 미탐 수정 (Phase 1 정규화 + Phase 2 데이터 fine-tune)

## 1. Background

### 1.1 Trigger issue
GPT-SoVITS로 본인 음성을 클로닝하여 합성한 `test-samples/fake/scenario_wonwoo_practicum.wav`(32kHz, 8.10s)를 v4 모델이 평균 fake 6.02%로 분류해 **REAL로 오판**. 학습 데이터에 GPT-SoVITS 합성물이 없고 시연 시나리오에 필요하므로 해결 필요.

### 1.2 Root cause (이전 디버깅 결과)
1. **Amplitude robustness collapse.** 모델 결정 경계가 입력 amplitude에 강하게 결합. 동일 음성을 ±6dB 조정하면 판정이 뒤집힘.
   - TTS2.wav: +0dB → 92.95%, -6dB → 7.40%, -12dB → 0.27%
   - real_01.wav: +0dB → 2.19%, +6dB → 70.79%, +12dB → 92.85%
2. **GPT-SoVITS 분포 외.** 학습 데이터는 edge-tts 다국어 + XTTS v2 클로닝만 포함.

시연 파일을 RMS 0.10~0.15로 정규화하면 v4 모델이 **그대로 73% fake 판정**. 즉 모델은 amplitude만 맞으면 GPT-SoVITS 시그니처를 식별할 수 있고, 본 issue는 추론 전처리 결함 + 데이터 분포 좁음의 결합이다.

### 1.3 기존 v5 작업 흔적
`model-training/data/train_v5.csv` (17,366 row), `val_v5.csv` (4,132 row)이 이미 존재한다 (사용자가 이전에 v5 데이터셋 준비를 시작한 상태). 본 작업은 이를 **베이스로 GPT-SoVITS 데이터를 추가**해서 `train_v4_5.csv`로 작성한다. 기존 v5 csv는 수정하지 않고 그대로 둔다.

### 1.4 아키텍처 교체를 취소한 이유
원래 AASIST2(arXiv 2309.08279) 도입을 검토했으나 Codex 독립 리뷰로 다음을 확인:
- AASIST2의 핵심 contribution은 wav2vec 2.0 XLS-R frontend + Res2Net + AM-Softmax + DCS + ALMFT. wav2vec 부분을 on-device 제약상 제거하면 논문 효과의 상당 부분이 사라지고, 결과가 ablation 없이 보장되지 않음.
- ONNX 호환성, ablation matrix, train/inference parity 등 추가 부담이 시연 일정 대비 과다.

따라서 본 PR은 **v4 ONNX를 유지**하고 (1) 추론 정규화 (2) GPT-SoVITS 데이터로 짧은 fine-tune 두 단계로 해결한다.

## 2. Goals / Non-Goals

### Goals
- 시연 4개 wav(scenario_wonwoo_practicum 포함)를 모두 fake로 정확 감지
- amplitude 변동(-12 ~ +12 dB)에서 모델 판정 안정화
- iPhone heldout real FP를 baseline 대비 2%p 이내로 유지 (regression band)
- GPT-SoVITS 시그니처를 학습 분포 내로 포함

### Non-Goals (명시 제외)
- AASIST2/AASIST3/Res2Net 등 모델 아키텍처 교체
- 통화 채널(VOICE_COMMUNICATION) 라이브 검증 — 별도 후속 작업
- 영어/다국어 확장 학습
- STT/피싱 키워드 매칭 로직 변경
- Android UI 변경

## 3. Phase 1 — Inference RMS 정규화

### 3.1 변경 위치
1. `model-server/api/routes/detect.py` — `_read_audio()` 반환 직전 정규화
2. `android-app/app/src/main/java/com/deepvoiceguard/app/inference/AudioPipeline.kt` — ONNX 입력 segment 직전 동일 로직

### 3.2 정규화 규칙
```
TARGET_RMS = 0.1
MAX_GAIN_DB = 30          # 너무 작은 신호 폭주 방지
MIN_RMS = 1e-4            # 이하는 near-silence로 간주, 정규화 skip
PEAK_LIMIT = 0.99         # 정규화 후 peak가 이를 넘으면 추가 attenuation

rms = sqrt(mean(x**2))
if rms < MIN_RMS:
    return x  # near-silence, 모델 통과 의미 없음 — 그대로 흘리기
gain = TARGET_RMS / rms
gain_db = 20 * log10(gain)
if gain_db > MAX_GAIN_DB:
    gain = 10**(MAX_GAIN_DB / 20)
y = x * gain
peak = max(abs(y))
if peak > PEAK_LIMIT:
    y = y * (PEAK_LIMIT / peak)
return y
```

### 3.3 단위 테스트
- `test_rms_normalize.py` (mac, pytest)
  - rms=0.05 → 정확히 rms=0.1 출력
  - rms=0.0001 → bypass (변경 없음)
  - rms=1.0 → 30dB cap → peak clip protection
  - rms=0.1 → 거의 변경 없음 (gain ≈ 1)
  - 12dB sine + 가산 노이즈 → rms target 도달 확인
- Android: 동일 케이스를 Kotlin unit test로 `AudioPipelineNormalizationTest.kt`

### 3.4 train/inference parity
**중요**: 학습에서도 같은 정규화가 적용되어야 score calibration이 일관됨. Phase 2 학습 단계에서 동일 함수를 dataloader에 통합. Phase 1만 적용된 동안에는 v4 baseline 메트릭도 정규화 path로 재측정 후 비교.

### 3.5 검증 (Phase 1 끝)
- 시연 4개 wav (scenario_wonwoo_practicum, Demo1/3/5/6 등) → 모두 fake 감지
- 추가 GPT-SoVITS 검증 4~5개 (Phase 2 합성에서 일부 회수) → fake 감지율 측정
- v4 baseline 정규화 path 재측정 (iPhone heldout / fake regression / mobile_fake_capture)
- Amplitude sweep -12 ~ +12 dB: 각 레벨에서 fake TPR ≥ 70%, real FPR ≤ 10%

### 3.6 Rollback (Phase 1만 적용된 상태에서 문제 발생)
단순 git revert. 모델 산출물 변경 없음.

## 4. Phase 2 — GPT-SoVITS 데이터로 v4 fine-tune

### 4.1 단계 A: GPT-SoVITS 합성 (Windows, 사용자 손)
산출물(mac에서 작성/commit):
- `tools/gpt_sovits_text_corpus.txt` — 300~500개 한국어 문장. 보이스피싱 시나리오 + 일상 발화 균형. 길이 분포 2~10초 예상.
- `tools/synth_gpt_sovits_batch.py` — Windows에서 실행할 batch 스크립트. GPT-SoVITS inference CLI 호출, output naming convention (`gptsovits_<speaker>_<idx>.wav`), 16kHz mono float32 저장.
- `docs/PHASE2_WINDOWS_PROMPT.md` — Windows 머신 Claude/Codex가 따를 작업 프롬프트:
  - 저장소 pull → 합성 스크립트 실행 → 결과 wav `model-training/data/gpt_sovits_v1/` 폴더에 저장 → commit/push
  - reference 음성: `test-samples/voice-clone-raw/phase0_ref_20260420_205403.wav` 와 `phase0_ref_20260422_clip014.wav` 둘 다 사용 (화자 다양성). batch 스크립트가 두 reference를 번갈아 사용하거나 사용자가 추가 reference 등록 가능하게 구성.

### 4.2 단계 B: 학습 (mac)
1. **데이터 통합**: 합성 wav pull → `model-training/data/train_v4_5.csv`, `val_v4_5.csv` 작성
   - 기존 train_v5(17,366) + 신규 GPT-SoVITS (예: 400) 합산
   - 50개 정도는 `test_gptsovits_holdout.csv`로 분리 (학습 미포함)
2. **Fine-tune 레시피** (`model-training/finetune_v4_5.py` 신규):
   - 시작점: v4-stable-2026-05-11 checkpoint (`aasist_best.pth`)
   - Optimizer: Adam, **lr = 5e-5** (catastrophic forgetting 방지로 작게)
   - Batch: 24 (MPS) / 64 (CUDA on Windows if used)
   - **Mixing**: 매 batch에서 real:fake 1:1 강제. 구현: weighted random sampler로 real/fake 두 풀에서 같은 수만큼 추출. real 풀이 더 크면 매 epoch random down-sample (epoch마다 다른 sample 보기).
   - **Augmentation**:
     - Gain: ±10dB uniform (실측 학습셋 RMS 분포 5~95 percentile 외부 ±6dB 정도)
     - RawBoost LnL only (Codex 권고 — 3 strategy 다 적용 안 함, 단순화)
     - Repeat-pad 유지 (CLAUDE.md 규약)
     - **모든 입력에 Phase 1과 동일한 RMS 정규화 적용** (train/inference parity)
   - **Loss**: CE (v4 그대로)
   - Max 15 epoch, early stop patience 3
   - **Best checkpoint 선택 기준**: val EER 절대치 우선. 만약 두 checkpoint의 EER 차이가 ≤ 0.5%p이면 real FPR 낮은 쪽 선택 (tie-breaker). real FPR이 baseline + 2%p 초과인 checkpoint는 후보에서 제외.
3. **머지 게이트**:
   - val EER ≤ baseline 또는 절대 EER 향상
   - real FPR ≤ baseline + 2%p
   - mobile_fake_capture ≥ 60% (demo gate)
   - 시연 4개 wav fake 감지
   - English TTS smoke 5/5 통과
4. **ONNX 변환** (`model-training/convert_to_onnx.py` 그대로, dynamo=False):
   - 변환 직후 ORT CPU + ORT Android (가능하면 emulator) load smoke test
   - PyTorch vs ONNX 출력 numeric parity 검증 (tolerance 1e-4)

### 4.3 단계 C: Android 배포
- `android-app/app/src/main/assets/aasist.onnx` 교체
- 모델 버전 string 업데이트 (`v4.5-gptsovits-2026-05-11` 같은 형식)
- 데모 4개 wav + 합성 holdout으로 빌드 검증

## 5. Validation Plan

### 5.1 Ablation rows (전부 같은 평가 데이터로)
| Row | 구성 | 측정 시점 |
|-----|------|----------|
| A | v4 baseline (정규화 없는 raw path, 기존 메트릭) | 작업 전 1회 |
| B | v4 + RMS norm only (Phase 1 끝) | Phase 1 완료 시 |
| C | v4 + RMS norm + GPT-SoVITS fine-tune (Phase 2 끝, 최종) | Phase 2 완료 시 |

### 5.2 메트릭 게이트 (C row가 통과해야 머지)
| 항목 | 합격 기준 |
|------|----------|
| iPhone heldout real FP | ≤ A_baseline + 2%p |
| Multi-TTS direct (test_fake_regression) | ≥ 95% |
| mobile_fake_capture (445개) | ≥ 60% (demo-driven) |
| English TTS regression (5개) | 5/5 smoke |
| Amplitude sweep -12 ~ +12 dB | 각 gain TPR ≥ 70% AND real FPR ≤ 10% AND variance ≤ 15 pp |
| 시연 4개 wav (scenario + Demo) | 전부 fake 감지 (avg ≥ 50%) |
| GPT-SoVITS holdout (~50개) | 정보 제공용 (gate 아님), 기대 ≥ 70% |

### 5.3 검증 스크립트
- `model-training/evaluate_v4_5.py` (신규): 위 7개 항목 전부 한 번에 측정, markdown 리포트 출력
- output: `docs/eval_reports/2026-05-XX-v4_5-eval.md`

## 6. Rollback / Fallback

### 6.1 산출물 버전화
- v4 ONNX 백업: `model-training/weights/v4-stable-2026-05-11/aasist.onnx` + sha256
- v4 checkpoint 백업: `model-training/weights/finetuned_v4_stable_2026-05-11/aasist_best.pth` + sha256 (이미 완료)
- v4.5 산출물: `model-training/weights/finetuned_v4_5/` + ONNX, 각 sha256 manifest 파일

### 6.2 Rollback 스크립트 (`tools/rollback_v4.sh`)
```bash
#!/usr/bin/env bash
# v4-stable로 복원. checksum 검증 포함.
set -e
SRC_PTH="model-training/weights/finetuned_v4_stable_2026-05-11/aasist_best.pth"
DST_PTH="model-training/weights/finetuned_v4/aasist_best.pth"
# ONNX는 v4-stable-2026-05-11 git 태그에서 추출
git show v4-stable-2026-05-11:android-app/app/src/main/assets/aasist.onnx > /tmp/aasist_v4.onnx
sha256sum -c manifests/v4_stable.sha256
cp /tmp/aasist_v4.onnx android-app/app/src/main/assets/aasist.onnx
cp "$SRC_PTH" "$DST_PTH"
echo "Rolled back to v4-stable-2026-05-11"
```

### 6.3 Fallback 시나리오
- **Phase 1 정규화에 의해 real FPR 악화** → 정규화 logic 조정 (target RMS 낮춤) 또는 Settings 토글 debug-only로 끄기
- **Phase 2 fine-tune 결과 미달** → v4-stable로 복원 + Phase 1 정규화만 유지하여 시연 (시연 파일은 정규화만으로도 73% 잡힘 확인됨)
- **ONNX export 호환성 깨짐** → 학습 중단, v4 ONNX 유지

### 6.4 자동 중단 트리거
- Loss NaN/Inf
- 3 epoch 연속 val EER 악화
- val EER이 v4 baseline(정규화 path) 대비 20% 이상 나쁨
- ONNX export 실패 (ORT CPU load 실패)

## 7. Deliverables (구체 파일)

### Phase 1
| 경로 | 신규/수정 | 설명 |
|------|----------|------|
| `model-server/api/routes/detect.py` | 수정 | `_read_audio` 후 RMS 정규화 호출 |
| `model-server/inference/normalization.py` | 신규 | 정규화 함수 단일 진실원 |
| `model-server/tests/test_normalization.py` | 신규 | 단위 테스트 |
| `android-app/app/src/main/java/.../inference/AudioPipeline.kt` | 수정 | ONNX 입력 직전 정규화 |
| `android-app/app/src/test/java/.../AudioPipelineNormalizationTest.kt` | 신규 | Kotlin 단위 테스트 |

### Phase 2
| 경로 | 신규/수정 | 설명 |
|------|----------|------|
| `tools/gpt_sovits_text_corpus.txt` | 신규 | 300~500 문장 |
| `tools/synth_gpt_sovits_batch.py` | 신규 | Windows 합성 batch |
| `docs/PHASE2_WINDOWS_PROMPT.md` | 신규 | Windows 측 Claude/Codex 프롬프트 |
| `model-training/data/gpt_sovits_v1/` | 신규(Windows pull 후) | 합성 wav 폴더 |
| `model-training/data/train_v4_5.csv` | 신규 | 통합 CSV |
| `model-training/data/val_v4_5.csv` | 신규 | |
| `model-training/data/test_gptsovits_holdout.csv` | 신규 | |
| `model-training/finetune_v4_5.py` | 신규 | fine-tune 스크립트 |
| `model-training/evaluate_v4_5.py` | 신규 | 검증 스크립트 |
| `model-training/weights/finetuned_v4_5/` | 신규(학습 후) | 새 checkpoint |
| `android-app/app/src/main/assets/aasist.onnx` | 수정 | v4.5 ONNX로 교체 |

### Rollback
| 경로 | 신규/수정 | 설명 |
|------|----------|------|
| `tools/rollback_v4.sh` | 신규 | 복원 스크립트 |
| `manifests/v4_stable.sha256` | 신규 | 체크섬 manifest |

## 8. Branch / Commit / PR 전략
- 작업 브랜치: `feature/aasist2-v5` → **`feature/v4_5-gptsovits-fix`로 rename**
- 베이스: `master` (a7b0548) + `v4-stable-2026-05-11` 태그
- Phase 1 PR: inference 정규화만 (작고 안전)
- Phase 2 PR: 데이터/학습/배포 (큰 PR, 메트릭 리포트 첨부)

## 9. Implementation Delegation
**사용자 결정**: 본 spec을 기반으로 한 코딩은 **Codex (gpt-5.5)**가 진행한다.
- spec 사용자 승인 후 → Codex에 본 문서 + 백업 절차 + 작업 브랜치 정보 전달
- Codex가 Phase 1을 먼저 구현/테스트/PR 생성 → 사용자/Claude 리뷰 → 머지
- 머지 후 Codex가 Phase 2 단계 A 산출물(텍스트 corpus, batch 스크립트, Windows 프롬프트) 작성
- 사용자가 Windows에서 합성 실행 → 결과 push
- Codex가 단계 B/C 진행 (fine-tune, ONNX 변환, Android 배포)

## 10. Open Questions / 시간 자원
- Codex 위임 시 작업 단위 크기: 단일 거대 task vs Phase 1/2 분리. **분리 권장** (Phase 1 산출물 본 후 Phase 2 진행 결정).
- 학습 하드웨어: M4 Pro MPS 우선. 학습 시간이 6시간 이상이면 Windows RTX 4080 활용 옵션 (Codex가 디바이스 선택 결정).
- Phase 1만 종료해도 시연 가능 — Phase 2는 정확도 개선 작업이지 시연 차단 요소 아님.
