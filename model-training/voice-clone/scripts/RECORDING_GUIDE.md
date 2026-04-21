# 녹음 가이드

`record_helper.py`는 Phase 0 빠른 레퍼런스 녹음과 Phase A 150문장 수집을 쉽게 진행하기 위한 터미널 도우미입니다. 한 문장씩 보여 주고, 각 문장을 48kHz / 24bit / mono WAV로 저장하며, 저장 직후 피크 레벨과 간단한 SNR을 검사합니다.

## 녹음 환경 권장사항

- 마이크는 입 정면에서 10~20cm 정도 거리를 두고 고정합니다.
- 키보드 타건, 선풍기, 창문 바깥 소리처럼 반복 잡음이 적은 방을 사용합니다.
- 한 세션은 10~20분 정도로 끊고, 중간에 물을 마시며 톤을 유지합니다.
- 문장 사이 호흡은 괜찮지만, 문장 시작 직전과 끝 직후 0.3초 정도는 조용히 유지하면 후처리가 편합니다.
- 자동 게인 보정보다는 고정 입력 게인을 권장합니다. 피크가 `-12dB ~ -6dB` 근처면 안정적입니다.

## 실행 방법

### 1. Phase 0 빠른 기준 음성

```bash
.venv-cloning/bin/python model-training/voice-clone/scripts/record_helper.py \
  --phase0 \
  --speaker hyohee
```

- 대본 없이 10초를 한 번 녹음합니다.
- 결과 파일은 `test-samples/voice-clone-raw/phase0_ref_<datetime>.wav`에 저장됩니다.

### 2. Phase A 150문장 녹음

```bash
.venv-cloning/bin/python model-training/voice-clone/scripts/record_helper.py \
  --script model-training/voice-clone/scripts/recording_script_ko.txt \
  --speaker hyohee
```

- 기본 대본은 `recording_script_ko.txt`입니다.
- 세션마다 `model-training/voice-clone/data/raw/<speaker>/session_<datetime>/`가 생성됩니다.
- 각 문장은 `clip_001.wav`, `clip_002.wav`처럼 저장되고, `session_log.json`에 상태가 누적됩니다.

### 3. 이어서 녹음

```bash
.venv-cloning/bin/python model-training/voice-clone/scripts/record_helper.py \
  --speaker hyohee \
  --resume
```

- 가장 최근 세션의 `session_log.json`을 읽고, 미완료 문장부터 다시 시작합니다.
- 특정 세션을 지정하려면 `--session session_20260420_103500` 같은 이름을 함께 넘기면 됩니다.

### 4. 입력 장치 확인 / 지정

```bash
.venv-cloning/bin/python model-training/voice-clone/scripts/record_helper.py --list-devices
```

```bash
.venv-cloning/bin/python model-training/voice-clone/scripts/record_helper.py \
  --speaker hyohee \
  --device 1
```

- `--device`에는 장치 번호 또는 이름 일부를 넣을 수 있습니다.
- 지정하지 않으면 입력 장치 목록을 보여 주고, 기본 장치를 선택할 수 있게 합니다.

## 단축키

- `[Enter]`: 현재 문장 녹음 시작
- 녹음 중 `[Enter]`: 녹음 종료
- `r`: 같은 문장을 다시 녹음
- `s`: 현재 문장을 스킵
- `b`: 이전 문장으로 이동
- `q`: 지금까지 저장하고 종료

## 자동 검사 항목

- 피크 레벨이 `-3dBFS`보다 높으면 `클리핑 위험` 경고를 띄웁니다.
- 피크 레벨이 `-24dBFS`보다 낮으면 `볼륨 낮음` 경고를 띄웁니다.
- 간단한 RMS 기반 추정 SNR이 `20dB` 미만이면 `잡음 감지` 경고를 띄웁니다.
- 문장 길이가 `2초 미만` 또는 `20초 초과`이면 길이 경고를 띄웁니다.

경고가 떠도 파일은 저장되지만, 가능하면 바로 재녹음하는 편이 좋습니다.

## 세션 구조

```text
model-training/voice-clone/data/raw/<speaker>/
└── session_<datetime>/
    ├── clip_001.wav
    ├── clip_002.wav
    ├── ...
    └── session_log.json
```

`session_log.json`에는 각 clip의 텍스트, 녹음 길이, peak/SNR, 경고 코드, `rec_status`가 들어 있습니다.

## 완료 후 다음 단계

녹음이 끝나면 전처리를 실행합니다.

```bash
.venv-cloning/bin/python model-training/voice-clone/preprocess_recordings.py \
  --speaker hyohee \
  --script model-training/voice-clone/scripts/recording_script_ko.txt
```

전처리 결과를 확인한 뒤 각 엔진별 fine-tuning 또는 zero-shot 경로로 넘기면 됩니다.

## 트러블슈팅

### macOS에서 마이크 권한 오류가 납니다

- `시스템 설정 > 개인정보 보호 및 보안 > 마이크`로 들어갑니다.
- 사용하는 터미널 앱(iTerm, Terminal, Warp 등)의 마이크 권한을 허용합니다.
- 권한을 바꾼 뒤에는 터미널 앱을 완전히 종료했다가 다시 실행합니다.

### 레벨이 너무 낮거나 너무 높습니다

- 너무 낮음: 마이크를 입 쪽으로 5cm 정도 더 가깝게 옮기거나 입력 게인을 올립니다.
- 너무 높음: 마이크를 멀리 두거나 입력 게인을 줄입니다.
- 목표는 일반 문장 읽기 기준 피크가 `-12dB ~ -6dB` 근처에 오도록 맞추는 것입니다.

### 잡음 경고가 계속 뜹니다

- 노트북 팬, 에어컨, 키보드 소리가 계속 들어오는지 확인합니다.
- 창문을 닫고, 가능한 한 동일한 자리에서 세션 전체를 유지합니다.
- 방음이 어렵다면 늦은 밤이나 새벽처럼 조용한 시간을 선택하는 편이 낫습니다.

### 이어하기가 예상과 다르게 동작합니다

- `session_log.json`이 있는 세션만 이어할 수 있습니다.
- `--resume --session <이름>`으로 세션을 명시하면 원하는 세션을 정확히 이어갈 수 있습니다.
