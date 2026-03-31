# 테스트 음성 녹음 대본

모델 입력이 5초(80,000 samples @ 16kHz)이므로, 각 대본은 **7~10초** 분량입니다.
자연스러운 속도로 읽어주세요.

---

## 일반 음성 (본인 녹음) — 기대 결과: SAFE

### 대본 1: 일상 대화
> 여보세요, 저 지금 퇴근하고 있는데요. 오늘 저녁에 같이 밥 먹을까요? 뭐 먹고 싶은 거 있으면 말해주세요.

### 대본 2: 업무 통화
> 네, 확인했습니다. 내일 오전 열 시에 회의 잡아놓을게요. 자료는 오늘 중으로 메일 보내드리겠습니다.

### 대본 3: 긴 발화
> 안녕하세요, 김효희입니다. 오늘 날씨가 정말 좋네요. 이번 주말에 시간 되시면 같이 카페에서 프로젝트 이야기 좀 하면 좋겠는데, 어떠세요?

---

## 딥페이크 음성 (TTS 생성) — 기대 결과: WARNING ~ DANGER

아래 텍스트를 TTS 서비스에 입력하세요.

### TTS용 텍스트 1
> 여보세요, 저 지금 퇴근하고 있는데요. 오늘 저녁에 같이 밥 먹을까요? 뭐 먹고 싶은 거 있으면 말해주세요.

### TTS용 텍스트 2
> 네, 확인했습니다. 내일 오전 열 시에 회의 잡아놓을게요. 자료는 오늘 중으로 메일 보내드리겠습니다.

### TTS용 텍스트 3 (영어)
> Hello, this is a test recording for the anti-deepfake voice detection system. The weather is nice today and I hope everything goes well with the project.

---

## TTS 서비스 추천

| 서비스 | 특징 | 탐지 가능성 |
|--------|------|-------------|
| macOS `say` 명령어 | 무료, 즉시 사용 | 낮음 (규칙 기반 TTS) |
| Google TTS | 무료 티어 있음 | 중간 |
| ElevenLabs | Voice Cloning 지원 | 높음 (가장 현실적인 deepfake) |
| Coqui TTS | 오픈소스 | 중간~높음 |

### macOS에서 빠르게 TTS 생성하기
```bash
# 영어 음성 생성
say -o tts_en.aiff "Hello, this is a test recording for the anti-deepfake voice detection system."

# AIFF → 16kHz WAV 변환 (ffmpeg 필요)
ffmpeg -i tts_en.aiff -ar 16000 -ac 1 tts_en.wav
```

### 녹음 파일 변환 (핸드폰 녹음 → 16kHz WAV)
```bash
ffmpeg -i my_recording.m4a -ar 16000 -ac 1 real_voice.wav
```

---

## 파일 정리

```
test-samples/
├── RECORDING_SCRIPT.md    ← 이 파일
├── real/                  ← 본인 녹음 파일 (16kHz WAV)
│   ├── real_01.wav
│   └── real_02.wav
└── fake/                  ← TTS 생성 파일 (16kHz WAV)
    ├── tts_google_01.wav
    └── tts_elevenlabs_01.wav
```
