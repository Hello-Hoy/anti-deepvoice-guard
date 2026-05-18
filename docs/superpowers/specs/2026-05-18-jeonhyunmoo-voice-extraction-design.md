# 전현무 단독 음성 추출 파이프라인 설계

- 작성일: 2026-05-18
- 상태: 승인됨 (브레인스토밍 완료)
- 접근법: A — Resemblyzer anchor 매칭 + gap-energy 클린니스

## 1. 목적

`voicebox-clone-workspace/[전현무계획3] ...` 폴더의 508개 `.m4a`(AAC 스테레오 44.1/48kHz, 오디오 총 ~18~20시간)에서 **전현무 단독 발화** 구간만 골라 **10~20초 깨끗한 클립**으로 잘라 하나의 폴더에 모은다. 예능 특성상 BGM·겹침·효과음이 상시 존재하므로 엄격한 클린니스 필터로 스튜디오 토크 위주 구간만 채택한다.

용도: anti-deepvoice-guard 프로젝트의 음성 클로닝 fake 데이터 소스 / 화자 코퍼스.

## 2. 확정된 요구사항 (브레인스토밍 결과)

| 항목 | 결정 |
|---|---|
| 전현무 식별 | 자동 부트스트랩 (메인 MC = 최다 파일 커버리지 화자), 사용자 청취 확인 게이트 |
| 처리 순서 | 파일럿 20~30개로 임계값 튜닝 → 승인 후 508개 전체 |
| 깨끗함 기준 | 엄격 (BGM·효과음 거의 없는 스튜디오 토크만), 소스 분리 사용 안 함 |
| 종료 기준 | 엄격 필터 통과분 전부, 품질 점수순 정렬 |
| 클립 길이 | 10~20초 |
| 출력 포맷 | 16kHz mono PCM16 wav (기존 워크스페이스 패턴 일치) |

## 3. 비목표 (Non-goals)

- pyannote/speechbrain 등 신규 의존성 설치 (Resemblyzer/librosa/ffmpeg만 사용)
- Demucs 등 소스 분리 (사용자가 엄격·원본 사용 선택)
- 기존 워크스페이스 스크립트/데이터 수정 (신규 파일만 추가)
- 화자 다이어리제이션 전체(전 출연자 분리) — 전현무 단일 타깃만

## 4. 아키텍처

단일 CLI 스크립트 `voicebox-clone-workspace/extract_jhm.py`, 3개 모드:
`--bootstrap` → `--pilot` → `--full`. 디코드 캐시와 manifest로 resumable.

격리된 함수 단위:

| 유닛 | 책임 | 입력 → 출력 |
|---|---|---|
| `decode_to_wav16k` | m4a → 16k mono wav 캐시 | m4a 경로 → 캐시 wav 경로 |
| `bootstrap_anchor` | 전현무 화자 클러스터 자동 발견 | 샘플 wav N개 → 후보 레퍼런스 클립 3개 + anchor 임베딩 |
| `cleanliness_gate` | 엄격 클린니스 판정 | 오디오 세그먼트 → (통과여부, 메트릭) |
| `extract_segments` | 화자매칭 + 턴 그룹핑 + 클립 선택 | wav + anchor → 클립 리스트 |
| `run_batch` | 오케스트레이션·manifest·재개 | 파일 리스트 → 출력 폴더 + manifest |

기존 검증된 패턴(`diarize_mp3_v2.py`의 Resemblyzer 임베딩 + 슬라이딩 윈도우 코사인 + top-N anchor 반복정제, `find_reference_segments.py`의 RMS/voiced 휴리스틱)을 다중 파일·엄격 클린니스로 일반화한다.

## 5. 데이터 흐름

```
508 m4a ──decode──> _cache/<hash>.wav (16k mono)
                          │
   [bootstrap] 샘플 30개 ──> Resemblyzer 윈도우 임베딩 ──> 클러스터링
                          └─> 최다 파일 커버리지 클러스터 = 전현무 후보 ──> 클립 3개 추출
                                                    │
                          [사용자 청취 확인] ◄───────┘
                                    │ 확인됨
                          anchor 임베딩 고정 (top-200 윈도우로 1회 반복정제)
                                    │
   각 파일 ──> 1.6s/0.4s 슬라이딩 윈도우 임베딩 ──> anchor 코사인 유사도
            ──> sim ≥ THR 윈도우 = 전현무 ──> 연속 윈도우 턴 그룹핑(짧은 gap 병합)
            ──> 턴 내 10~20s 후보 ──> [클린니스 게이트] ──> 통과분만
            ──> 품질점수 = mean_sim × √dur × clean_score
                                    │
   jeonhyunmoo_clean/ <- 통과 클립 전부 (점수순 정렬) + manifest.json
```

## 6. Anchor 부트스트랩 (전현무 자동 식별)

- 짧은 클립(예고편/쇼츠) 중 30개 샘플 선정 → 각 파일 슬라이딩 윈도우 임베딩 수집
- 전체 윈도우 임베딩 정규화 후 클러스터링(KMeans k≈6). **전현무는 거의 모든 클립에 등장하는 메인 MC** → 단순 최다 윈도우가 아니라 **"가장 많은 서로 다른 소스 파일에 걸쳐 분포하는 클러스터"**를 전현무로 선정 (내레이션/특정 게스트 편향 방지)
- 해당 클러스터 centroid에 가장 가까운, 서로 다른 3개 파일에서 대표 클립(각 8~12s) 추출 → `_anchor_candidates/`
- **사용자가 3개 클립을 듣고 "전현무 맞음" 확인** (확인 게이트). 부정 시 차순위 클러스터 제시
- 확인 후: 클러스터 + 확인 클립 평균을 anchor로, 이후 전체 샘플에서 top-200 고유사 윈도우로 1회 반복정제. `_anchor/anchor.npy`로 저장하여 pilot/full에서 재사용

## 7. 엄격 클린니스 게이트 (핵심)

세그먼트가 **모든** 조건을 통과해야 채택:

1. **화자 신뢰도**: anchor 코사인 sim ≥ `SIM_THR` (파일럿서 튜닝, 초기 0.80). 겹침 구간은 임베딩이 화자 사이로 흐려져 sim 하락 → 자연 배제
2. **gap-energy (BGM 판별, 핵심)**: 세그먼트 내 pause 프레임 검출(RMS < voiced_thr) → pause 프레임 평균 RMS가 전역 노이즈플로어의 작은 배수(초기 4×) 이하여야 통과. BGM이 깔리면 pause에도 음악 에너지 잔존 → 탈락. 반대로 pause가 거의 없어도(연속 에너지 = 음악/내레이션 베드 의심) 탈락
3. **voiced ratio** ≥ 0.7, **최장 무음** ≤ 1.5s
4. **클리핑**: peak ≤ 0.95
5. **스펙트럼 보조**: spectral_flatness 중앙값으로 tonal(음악/효과음) 의심 구간 컷
6. **길이**: 턴에서 10~20s 서브클립 절취 (20s 초과 턴은 중앙 절취)

초기 임계값은 파일럿 청취 결과로 튜닝한다.

## 8. 에러 처리 / 재개

- 디코드 실패 m4a → manifest `errors[]`에 명시 기록, 배치 계속 (silent skip 금지)
- 전현무 윈도우 0개 파일 → 정상(출력 없음), 에러 아님
- resumable: 디코드 wav는 `_cache/`에 소스 mtime+size 해시로 캐시, manifest에 처리완료 소스 기록 → 재실행 시 스킵
- 부트스트랩 anchor는 `_anchor/anchor.npy` 저장 → 재사용, 강제 재부트스트랩은 `--rebootstrap` 플래그

## 9. 검증 방법

- **부트스트랩 확인 게이트** (사용자 청취) — 화자 정확성 1차 검증
- **파일럿 게이트**: 20~30파일 처리 후 산출 클립 중 무작위 6~8개 사용자 청취 → `SIM_THR`·gap-energy 임계 튜닝 → 재확인 후 전체 진행
- 전체 실행 후 통계 리포트: 총 클립 수·총 분량·파일당 수율·sim 분포, 무작위 spot-check 5개
- 회귀 없음: 기존 스크립트/데이터 무변경, 신규 파일만 추가

## 10. 산출물 스펙

- 폴더: `voicebox-clone-workspace/jeonhyunmoo_clean/`
- 포맷: 16kHz mono PCM16 wav, 10~20s
- 파일명: `jhm_{src순번:03d}_{t0}s_{dur}s_sim{NN}.wav`
- `manifest.json`: 소스파일·시작·길이·sim·gap_energy·voiced_ratio·spectral_flatness·clean_score·총점, `errors[]`, 통계 요약

## 11. 런타임 예상

오디오 ~18~20h, 1.6s/0.4s 윈도우 ≈ ~18만 윈도우. Resemblyzer CPU 임베딩 ≈ 수 시간(전체). 디코드는 ffmpeg I/O 바운드로 빠름. 파일럿(30파일)은 수십 분 내로 위험 선제거.

## 12. 작업 산출 파일 목록

신규:
- `voicebox-clone-workspace/extract_jhm.py` (CLI, 전 유닛 포함)
- `voicebox-clone-workspace/jeonhyunmoo_clean/` (출력, gitignore 대상 — 대용량)
- `voicebox-clone-workspace/_cache/`, `_anchor/`, `_anchor_candidates/` (중간산물, gitignore)

수정: 없음 (기존 스크립트/데이터 무변경)
