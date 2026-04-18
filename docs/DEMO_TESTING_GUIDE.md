# Demo 테스트 가이드 (컴퓨터 시연)

Android Studio 에뮬레이터에서 Demo 탭으로 전체 파이프라인을 시연하는 방법.

## 사전 준비 (다른 컴퓨터에서)

### 1. 필수 설치

| 도구 | 버전 |
|------|------|
| JDK | 17 (Temurin 또는 OpenJDK 17) |
| Android Studio | Ladybug 이상 |
| Android SDK | API 34 (Android 14) |
| Kotlin | 1.9.x (Gradle이 자동 설치) |

### 2. 저장소 클론

```bash
git clone https://github.com/Hello-Hoy/anti-deepvoice-guard.git
cd anti-deepvoice-guard
```

### 3. JDK 17 환경변수 (Mac 기준)

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

또는 Homebrew OpenJDK 17:
```bash
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.18/libexec/openjdk.jdk/Contents/Home
```

JDK 24 이상은 Android Gradle Plugin과 호환 안 됨 (빌드 실패).

### 4. 에뮬레이터 생성

Android Studio → **Device Manager → Create Device**

- **Device**: Pixel 7
- **System Image**: Android 14 (API 34), **Google Play** ARM 64 v8a
- Korean STT 사용을 위해 Google Play 이미지 권장 (하지만 Demo 모드만 쓸 거면 필수 아님)

## Demo 시연 순서

### 1. 빌드 + 실행

Android Studio에서:
1. **File → Open** → `anti-deepvoice-guard/android-app/` 선택
2. Gradle sync 자동 진행 (1~3분)
3. 에뮬레이터 시작
4. 상단 녹색 **▶ Run 'app'** 클릭

처음 빌드 시 5~8분 소요. 이후는 30초~1분.

### 2. 앱 실행 화면

앱이 자동으로 뜨면 **Home 탭**이 보입니다:
- 제목: "Anti-DeepVoice Guard"
- 상태: "Monitoring Stopped"
- 큰 파란색 **START** 버튼

하단 **탭 5개**: Home / **Demo** / History / Settings / About

### 3. Demo 탭 진입

하단 **Demo** 탭 클릭 → 6개 시나리오 카드가 보입니다:

| # | 제목 | 예상 결과 |
|---|------|---------|
| 1 | 일상 통화 | SAFE |
| 2 | TTS 일상 대화 | DANGER |
| 3 | 실제 사람 피싱 | WARNING |
| 4 | TTS 피싱 | CRITICAL |
| 5 | 은행 정상 안내 | SAFE |
| 6 | 보험 사기 | WARNING |

### 4. 시나리오 분석

각 카드의 **분석** 버튼 클릭:
1. 실제 오디오가 **스피커로 재생** (약 10~15초)
2. **파형 애니메이션** 바가 움직임
3. 재생 중 프로그레스 바 채워짐
4. 재생 완료 후 결과 카드가 뜸:
   - **통합 위협 레벨** (SAFE / WARNING / DANGER / CRITICAL)
   - 딥보이스 점수 (%)
   - 피싱 점수 (%)
   - 감지된 피싱 키워드 (빨간색)
   - 전사 텍스트 (타이핑 효과)
   - 피싱 키워드가 전사에 빨간색 하이라이트

### 5. 체크리스트

시연 전 확인:

- [ ] #1: SAFE, 딥보이스 ~40%
- [ ] #2: DANGER, 딥보이스 ~95%
- [ ] #3: WARNING, 피싱 키워드 "검찰", "수사관", "계좌이체", "안전한 계좌", "체포", "범죄", "수사" 등 빨갛게
- [ ] #4: **CRITICAL** (경진대회 핵심 시연 포인트), 딥보이스 ~89% + 피싱 100%
- [ ] #5: SAFE (오탐 없음, 오탐 방지 시연)
- [ ] #6: WARNING, 피싱 키워드 "금융감독원", "오늘 안에", "불이익" 등

6개 모두 예상과 일치해야 합니다.

## 문제 해결

### "Gradle sync failed" / "Unsupported Java version"
→ JDK 17 환경변수 확인. `java -version`이 17.x.x로 나와야 함.

### "Build failed: ... AndroidManifest.xml"
→ Android Studio에서 **File → Sync Project with Gradle Files** 다시 실행.

### Demo 탭이 안 보임
→ 6개 WAV 중 **하나라도 있어야** Demo 탭이 활성화됨. `android-app/app/src/main/assets/demo/`에 `demo_01.wav ~ demo_06.wav`가 있는지 확인.

### 오디오 재생이 안 들림
→ 에뮬레이터 볼륨 확인 (에뮬레이터 우측 툴바의 볼륨 버튼). 또는 Mac 시스템 볼륨.

### 분석 결과가 예상과 다름
→ 같은 시나리오를 2~3번 반복 분석 (캐시 워밍업 목적). 계속 다르게 나오면 `android-app/app/src/main/assets/aasist.onnx` 파일 크기 확인 (644KB여야 정상).

### 앱이 실행 즉시 크래시
→ Logcat에서 `tag:AndroidRuntime` 필터 → 빨간 **FATAL EXCEPTION** 스택트레이스 전체를 이슈로 보고.

## 발표 시 팁

- **프로젝터 해상도**: 에뮬레이터 창을 **1:1 스케일**로 설정하고 화면을 **녹화**해두기 (실시간 시연 실패 대비).
- **시나리오 순서**: #1 (SAFE) → #4 (CRITICAL) → #3 (WARNING, 피싱 키워드 하이라이트 강조) → #5 (오탐 없음 증명) 순서가 임팩트 있음.
- **소요 시간**: 6개 전체 ~2~3분 (각 시나리오 재생 15초 + 설명 15초).

## 기술 세부사항

Demo 파이프라인은 `DemoAnalysisPipeline.kt`에서 다음을 수행:

1. WAV asset 로드 → 16kHz mono FloatArray
2. 4초 sliding window 세그먼트 분할 (50% overlap)
3. 각 세그먼트 → AASIST ONNX 추론 → 평균 fake/real 점수
4. 사전 전사본 로드 → PhishingKeywordDetector
5. CombinedThreatAggregator 5단계 결합

Demo WAV는 미리 narrowband band-pass + RMS normalize되어 저장됨 (`tools/preprocess_demo_wav.py`). 이는 iPhone 마이크 녹음 특성을 AASIST 학습 분포에 맞추기 위함.
