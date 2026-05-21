# 전현무 GPT-SoVITS 학습 데이터셋 + 이식형 파이프라인 — 설계

- 날짜: 2026-05-21
- 브랜치: feature/jeonhyunmoo-voice-extraction
- 관련: [[2026-05-18-jeonhyunmoo-voice-extraction-design]] (추출 파이프라인), 프랙티컴 경진대회 데이터 파이프라인

## 1. 목표 & 산출물

`voicebox-clone-workspace/전현무음성/`의 m4a(109개)에서 **전현무 단독 깨끗한 발화**를 추출하여
Windows GPT-SoVITS 학습에 바로 쓸 수 있는 데이터셋으로 패키징한다.

산출물은 두 가지(사용자 확정):
1. **데이터셋** — 개별 wav 세그먼트(32kHz) + GPT-SoVITS `.list` 전사 파일
2. **이식형 재실행 스크립트** — Windows(CUDA)에서 동일 추출을 재실행 가능

방향 결정(사용자 확정):
- **순도 최우선** (양보다 전현무 단독 확신 발화) — GPT-SoVITS는 소량으로도 fine-tune 되므로 클론 음색 안정성 우선
- **순도 검증 = 자동 고임계 + montage 귀확인 후 reject 반영** (반자동)
- **export SR = 32kHz** (GPT-SoVITS 합성 타깃 SR)

## 2. 접근

기존 `jhm/` 패키지(검증·테스트 완료: decode→Demucs 보컬분리→Silero VAD→resemblyzer 임베딩→anchor 코사인 임계)를
**재사용·확장**한다. 핵심 난제(보컬분리·발화분할·anchor)는 이미 작동하므로 신규 코드는 GPT-SoVITS 어댑테이션에 한정한다.

## 3. 디렉터리 구조

```
voicebox-clone-workspace/
├── jhm/
│   ├── core.py               # 기존(재사용). 일부 함수 export-SR 인자 추가
│   ├── device.py             # [신규] pick_device(): cuda>mps>cpu
│   └── gptsovits.py          # [신규] 세그먼트 export + 전사 + .list 빌더 + montage/reject
├── build_gptsovits_dataset.py# [신규] CLI: probe / rebuild-anchor / extract / montage / finalize
└── jhm_work/gptsovits/
    ├── staging/
    │   ├── segments/jhm_0001.wav …   # 채택 발화 개별 wav (32k)
    │   ├── segments.json             # [{id, source, start_s, end_s, dur, sim}]
    │   ├── preview/montage_all.wav    # sim 내림차순 미리듣기
    │   ├── preview/montage_<source>.wav
    │   ├── preview/timeline.txt       # montage 내 seg id ↔ 시각 매핑
    │   └── reject.txt                 # 사용자가 제외할 seg id 기입(주석 템플릿)
    └── dataset/                       # ★최종 — 이 폴더째 Windows 이동
        ├── wavs/jhm_0001.wav …        # 승인+전사된 세그먼트 (32k)
        ├── jhm.list                   # "wavs/jhm_0001.wav|jhm|ko|전사텍스트"
        ├── relocate_list.py           # Windows 절대경로 1회 변환
        ├── requirements.txt
        └── README.md
```

## 4. CLI 단계 (`build_gptsovits_dataset.py`)

각 단계는 독립 실행 가능(이전 단계 산출물을 입력으로 받음).

1. **`probe [--n N] [--thresholds ...]`**
   샘플 N개 파일 × 임계값 후보로 수율(채택 분량)·sim 분포를 리포트. 순도 임계 T 캘리브레이션(실측 우선).
   파일 출력 없이 콘솔 리포트.
2. **`rebuild-anchor`**
   클린 4-ref(`montage_cluster_01/05/07/16.wav`, cluster_08 제외)의 resemblyzer 임베딩 평균 → 정규화 →
   `jhm_work/anchor/anchor.npy` 갱신. (진단상 cluster_08이 가장 이질적이므로 제외하여 순도 극대화)
3. **`extract --threshold T [--limit K]`**
   전체 소스(또는 K개)에 대해: decode→보컬분리(캐시)→VAD→임베딩→sim≥T 채택.
   채택 발화를 **개별 wav(32k)** 로 `staging/segments/`에 저장 + `segments.json` 기록.
4. **`montage`**
   `segments.json` 기준으로 미리듣기 wav 생성(전체 sim 내림차순 + 소스별) + `timeline.txt` +
   `reject.txt` 템플릿 출력. 사용자가 듣고 제외할 seg id를 reject.txt에 적는다.
5. **`finalize`**
   reject.txt 반영 → 남은 세그먼트를 faster-whisper(ko)로 전사 → 빈 전사 제외 →
   `dataset/wavs/`로 복사 + `jhm.list` + `relocate_list.py` + `requirements.txt` + `README.md` 생성.

## 5. 핵심 기술 결정

- **anchor = 클린 4-ref 평균** (cluster_08 제외). 세션 진단에서 cluster_08이 상호유사도 0.799~0.878로 가장 이질적.
- **분석 16k / export 32k 분리.**
  - VAD·임베딩은 기존대로 16k 보컬에서 수행(테스트된 경로 유지).
  - `separate_vocals`에 export-SR 옵션을 추가하여 demucs 내부 44.1k 보컬을 **32k로도 캐시**.
  - VAD 타임스탬프(16k 샘플)는 초 단위로 변환 후 32k 보컬에서 슬라이스 → 32k 세그먼트 export.
  - 근거: 소스 m4a는 대역제한적이나 GPT-SoVITS 합성 타깃이 32k이므로 16k 학습 시 음질 손해.
- **전사 = faster-whisper `large-v3`, language="ko".** Mac=CPU(int8), Windows=CUDA(float16). (ctranslate2는 MPS 미지원)
- **`.list` 포맷 = `상대경로|jhm|ko|전사`.** dataset 루트 기준 상대경로로 기록.
  `relocate_list.py --base <Windows 절대 dir>`로 1회 절대경로 변환.
- **순도 게이트(반자동).** 자동 sim≥T + montage 귀확인 → reject.txt로 수동 제외. 완전자동 아님.
- **device 자동감지.** `pick_device()`: cuda 있으면 cuda, 없으면 mps, 없으면 cpu. demucs·resemblyzer에 적용.
  (resemblyzer 인코더는 경량이라 cpu 유지 가능 — 옵션화)

## 6. 에러 핸들링

- 파일 단위 try/except — 한 m4a 실패가 전체 추출을 중단시키지 않음(스킵 카운트 로그).
- 빈 전사·과단(<min_dur) 발화 제외.
- 조용한 폴백 금지 — anchor.npy 없음/모델 로드 실패 등은 명시적 RuntimeError.

## 7. 테스트 (TDD)

순수 함수 우선, 합성 데이터로 단위테스트:
- `pick_device()` — 환경 모킹별 반환값.
- `.list` 빌더 — manifest→포맷 문자열, 빈 전사 제외, 경로 상대화.
- reject 필터 — reject.txt 파싱(주석·공백 무시) + 세그먼트 제외.
- segments.json 직렬화/역직렬화 라운드트립.
- 32k 슬라이스 타임스탬프 변환(16k 샘플→초→32k 샘플) 경계값.
demucs/VAD/임베딩 통합부는 기존 jhm 테스트 유지(실오디오 의존부는 신규 단위테스트 대상 아님).

## 8. 범위 제외 (YAGNI)

- GPT-SoVITS 학습 실행·하이퍼파라미터·WebUI 자동화(Windows에서 사용자 수행).
- 정식 화자분리(diarization) 도입 — anchor 유사도 게이트로 충분.
- 16k/24k export 옵션(32k 단일).

## 9. 미해결/리스크

- 순도 임계 T는 `probe` 실측 후 확정(현재 미정). 109개 m4a의 32k 클린 수율은 미지수.
- 변별: anchor 유사도가 화자 경계에 완벽치 않을 수 있음 → montage 귀확인으로 보완(반자동).
- faster-whisper Mac CPU 전사 속도(대량 시 느릴 수 있음) — Windows CUDA에서 재전사 가능.
