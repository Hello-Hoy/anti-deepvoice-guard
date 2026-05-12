# Phase 1 정규화 시도 → 롤백 결정 (2026-05-12)

## 요약
RMS 정규화를 추론 경로(server `detect.py` + Android `AudioPipeline.kt`)에만 적용하는 Phase 1 작업을 완료했으나, **iPhone heldout real 정확도가 66.7% → 13.8%로 폭락**하여 시연에서 일반 음성을 fake로 오탐하는 회귀가 확인되어 **롤백**.

## 시도한 작업 (모두 backup branch 에 보존)
브랜치: `backup/phase1-attempt-2026-05-12` (7 commit, 마지막 32a3fc6)
- `model-server/inference/normalization.py` — `normalize_rms` (TARGET_RMS=0.1, MAX_GAIN_DB=30, MIN_RMS=1e-4, PEAK_LIMIT=0.99)
- `model-server/api/routes/detect.py` — `_read_audio` 직후 정규화 호출
- `android-app/.../service/AudioNormalization.kt` — Python 모듈 미러
- `android-app/.../service/AudioPipeline.kt` — `inferenceEngine.detect()` 직전 정규화 호출
- Python 단위 테스트 8개, Kotlin 단위 테스트 6개 (Python/Kotlin parity smoke 포함) 전부 통과
- `model-training/measure_normalization_baseline.py` — ablation 측정 스크립트

## 측정 결과 (A: raw, B: RMS norm)

| Holdout | A (v4 baseline, no norm) | B (v4 + normalize_rms) | 변화 |
|---------|--------------------------|------------------------|------|
| iPhone heldout real (87 wav) | 66.7% acc (58/87) | **13.8% acc (12/87)** | **−52.9 pp** |
| Fake regression (5 wav) | 100% (5/5) | 100% (5/5) | 0 |
| Speaker holdout jeong real (50 wav) | 48.0% acc (24/50) | **24.0% acc (12/50)** | **−24.0 pp** |

시연 wav fake 감지 (정규화 적용 후):
- scenario_wonwoo_practicum: 59.20% ✅
- TTS1/2, fake_01/02: 98~99% ✅
- 시연 자체는 통과하지만 일반 통화의 real 음성은 87개 중 75개 fake 오탐

## 근본 원인
v4 모델은 raw amplitude 분포로 학습되었다. iPhone 실제 녹음은 저-RMS (~0.03~0.06). 모델은 그 amplitude 대역을 "real"로 학습. 정규화로 RMS 0.1로 끌어올리면 모델의 "큰 amplitude = fake" 결정 경계가 활성화되어 real을 fake로 오탐.

즉 **train/inference mismatch**. spec 단계의 Codex 독립 리뷰에서 이미 지적된 사항:
> "RMS-target normalization at inference is a major behavior change and must be trained into the model path, not bolted onto server/Android only. Train and validate with the same normalization."

원본 spec(`docs/superpowers/specs/2026-05-11-gptsovits-miss-fix-design.md`) Section 3.4에 "Phase 2 학습 단계에서 동일 함수를 dataloader에 통합"한다고 명시했지만, **Phase 1 단독 적용 시점에 calibration이 깨지는 정도를 과소평가**.

## 결정
- **이 시도는 전부 롤백** (메인 작업 브랜치 `feature/v4_5-gptsovits-fix`를 `c87acf5` plan commit으로 reset)
- spec/plan 문서는 그대로 유지 (작업 기록으로 가치 있음)
- Phase 1 코드는 `backup/phase1-attempt-2026-05-12` 에 보존 — 미래에 학습 단계와 함께 도입하려면 cherry-pick 가능
- v4 production 상태(`v4-stable-2026-05-11` tag, `aasist.onnx` master 그대로)는 무손상

## 다음 방향
사용자 결정 대기 중. 후보:
1. **학습부터 정규화 포함하는 새 접근** — Phase 2 단독으로 시작 (data augmentation + normalization을 학습 path 에 통합 후 추론에 한꺼번에 적용)
2. **정규화 외 다른 amplitude 대응** — 데모용 사전 정규화(파일 자체를 정규화해서 시연), 또는 시연 경로(`DemoAnalysisPipeline`)에만 적용
3. **시연 파일 자체 amplitude 사전 조정** — wav 파일에 직접 정규화 적용 후 저장, 모델/앱은 그대로

## 참조
- spec: `docs/superpowers/specs/2026-05-11-gptsovits-miss-fix-design.md`
- plan: `docs/superpowers/plans/2026-05-11-gptsovits-miss-fix.md`
- backup branch: `backup/phase1-attempt-2026-05-12`
- v4 stable: tag `v4-stable-2026-05-11`
