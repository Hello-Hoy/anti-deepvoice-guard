# 팀원 온보딩 가이드 — Android Studio로 데모 실행하기

처음 합류한 팀원이 **환경 설치 → 저장소 받기 → 빌드 → 데모(특히 데모11 양정윤 사칭) 재생**까지 직접 할 수 있도록 정리한 문서입니다. macOS / Windows 공통이며, OS별 차이는 따로 표시했습니다.

> 이미 환경이 있는 분은 [3. 저장소 받기](#3-저장소-받기)부터 보세요.
> 데모 시연 체크리스트는 [`DEMO_TESTING_GUIDE.md`](DEMO_TESTING_GUIDE.md), 실기기 테스트는 [`REAL_DEVICE_TESTING.md`](REAL_DEVICE_TESTING.md), 피싱 점수 원리는 [`STT_PHISHING_SCORE_EXPLANATION.md`](STT_PHISHING_SCORE_EXPLANATION.md) 참조.

---

## 1. 한눈에 보는 요구 사양

| 항목 | 값 | 비고 |
|---|---|---|
| **JDK** | **17** (Temurin / OpenJDK 17) | ⚠️ JDK 24+ 는 빌드 실패. 반드시 17 |
| Android Studio | Ladybug(2024.2) 이상 | Gradle/AGP가 함께 들어옴 |
| Android SDK | **API 35** (Android 15) | `compileSdk=35`, `targetSdk=35` |
| 최소 실행 기기 | API 26 (Android 8.0) 이상 | `minSdk=26` |
| Gradle / AGP / Kotlin | 8.13 / 8.13.2 / 2.1.0 | Gradle Wrapper가 자동 설치 |

ONNX 모델(`aasist.onnx`, `silero_vad.onnx`)과 데모 음성은 **저장소에 포함**되어 있어 별도 다운로드가 필요 없습니다.

---

## 2. 사전 설치

### 2-1. Git
- macOS: `git --version` (없으면 `xcode-select --install`)
- Windows: <https://git-scm.com> 설치

### 2-2. JDK 17
Android Studio 내장 JBR(17)을 써도 되지만, 커맨드라인 빌드까지 하려면 별도 설치 권장.

- **macOS (Homebrew)**: `brew install --cask temurin@17`
- **Windows**: [Adoptium Temurin 17](https://adoptium.net/temurin/releases/?version=17) MSI 설치

확인:
```bash
java -version   # 17.x.x 가 나와야 함
```

### 2-3. Android Studio
<https://developer.android.com/studio> 에서 설치 → 첫 실행 시 SDK 설치 마법사에서 **Android SDK Platform 35** 와 **Android SDK Build-Tools** 포함.

---

## 3. 저장소 받기

```bash
git clone https://github.com/Hello-Hoy/anti-deepvoice-guard.git
cd anti-deepvoice-guard
```

`master` 브랜치에 데모11까지 모두 포함되어 있습니다(별도 checkout 불필요).

---

## 4. Android Studio로 프로젝트 열기

1. Android Studio → **File → Open**
2. ⚠️ **`anti-deepvoice-guard/android-app/` 폴더를 선택** (저장소 최상위 루트가 아님 — 루트를 열면 Gradle 인식 실패)
3. **Gradle Sync** 자동 시작 (처음엔 의존성 다운로드로 1~3분)

### Gradle JDK를 17로 지정
**Settings/Preferences → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK** → `17` 선택 (없으면 "Download JDK"로 Temurin 17 받기).

> Sync 실패 / "Unsupported Java version" 오류 → 거의 100% JDK 문제입니다. 위 설정을 17로 맞추세요.

---

## 5. 실행 대상 준비 — 에뮬레이터 또는 실기기

### 5-A. 에뮬레이터
1. **Device Manager → Create Device** → **Pixel 7**
2. System Image: **API 35**(또는 34) — Google Play 이미지 권장
3. Finish

### ⚠️ 5-B. 가장 흔한 함정 — "소리가 안 나요"

에뮬레이터가 **Android Studio 안에 도킹된 "Running Devices" 창**(임베디드 모드)으로 뜨면 **호스트 PC 스피커로 오디오가 출력되지 않습니다.** 데모 음성이 무음으로 들립니다. (코드 문제가 아님)

**해결 — 에뮬레이터를 독립(standalone) 창으로 실행:**
1. **Settings/Preferences → Tools → Emulator**
2. **"Launch in the Running Devices tool window"** 체크 **해제**
3. 에뮬레이터를 다시 시작 → 별도 창으로 뜨면 소리가 정상 출력됩니다.

> 그래도 안 들리면: ① 에뮬레이터 우측 툴바 볼륨 버튼, ② PC 시스템 볼륨, ③ (가능하면) **실기기 사용**이 가장 확실합니다.

### 5-C. 실기기 (가장 확실)
USB 연결 → 기기에서 **개발자 옵션 → USB 디버깅** 활성화 → 상단 기기 목록에서 선택. 자세한 내용은 [`REAL_DEVICE_TESTING.md`](REAL_DEVICE_TESTING.md).

---

## 6. 빌드 & 실행

상단 툴바에서 실행 대상(에뮬레이터/기기) 선택 후 녹색 **▶ Run 'app'** 클릭.

- 첫 빌드: 5~8분, 이후: 30초~1분
- 앱이 뜨면 하단에 탭 5개: **Home / Demo / History / Settings / About**

---

## 7. 데모 실행 — 데모11 (양정윤 사칭) 중심

1. 하단 **Demo** 탭 클릭
2. **#11 「양정윤 사칭 지원금 요구(AI)」** 카드의 **분석** 버튼
3. 분석 선행(PREPARING 스피너) 후 음성이 재생되며 5개 지표가 **실시간 스트리밍**됩니다.

### 기대 결과

| 지표 | 동작 |
|---|---|
| **딥보이스 점수** | 합성 음성 구간에서 상승 → 괄호 등급(SAFE→CAUTION→WARNING)이 점수에 맞춰 함께 변함 |
| **피싱 점수** | "100만 원" 등장 시 **10%(LOW)** → "계좌" 등장 시 **35%(MEDIUM)** 로 누적 |
| **통합 위협도** | 딥보이스·피싱 결합 결과(WARNING 수준) — 표시 점수 기준으로 일관되게 표시 |
| **전사 텍스트** | 오디오 진행에 맞춰 한 글자씩 표시, 피싱 키워드("계좌") 빨간 하이라이트 |

> "키워드는 '계좌' 하나인데 왜 10%→35%로 오르나?"에 대한 설명은 [`STT_PHISHING_SCORE_EXPLANATION.md`](STT_PHISHING_SCORE_EXPLANATION.md)에 정리되어 있습니다. (요약: 점수 = 키워드 가중치 + 카테고리 보너스 + 금액 패턴 보너스의 합)

다른 시나리오(#1~#10) 기대값은 [`DEMO_TESTING_GUIDE.md`](DEMO_TESTING_GUIDE.md) 참조.

---

## 8. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| Gradle sync 실패, "Unsupported Java version" | Gradle JDK를 **17**로 (4장) |
| "Build failed: AndroidManifest.xml" | **File → Sync Project with Gradle Files** 재실행 |
| **데모 음성 무음** | 에뮬레이터를 **독립 창**으로 실행(5-B). 또는 실기기 사용 |
| Demo 탭이 안 보임/비활성 | `android-app/app/src/main/assets/demo/` 에 `demo_*.wav` 가 있는지 확인 |
| 분석 결과가 매번 다름 | 같은 시나리오 2~3번 반복(워밍업). 계속 이상하면 `assets/aasist.onnx` 크기 확인(약 1.8MB) |
| 앱 실행 즉시 크래시 | **Logcat** → `tag:AndroidRuntime` 필터 → FATAL EXCEPTION 스택트레이스 전체 공유 |

---

## 9. (선택) 데모 음성 직접 만들기

GPT-SoVITS 등으로 만든 원본 wav를 데모용 16kHz mono로 변환하는 스크립트:

```bash
python tools/make_yjy_demo.py   # 예: yjy_money_ver.wav → demo_11.wav (16kHz mono, peak 0.97)
```

> ⚠️ 클론 음성 데모(데모08~11)는 **narrowband(협대역) 전처리를 적용하지 않습니다.** raw 16kHz를 그대로 사용해야 딥보이스 점수가 정상적으로 나옵니다(협대역 적용 시 점수 왜곡). 새 데모를 추가할 때 주의하세요.

---

## 부록 — 빠른 시작 요약

```text
1. JDK 17 설치 + Android Studio 설치
2. git clone https://github.com/Hello-Hoy/anti-deepvoice-guard.git
3. Android Studio → Open → anti-deepvoice-guard/android-app/
4. Gradle JDK = 17 확인 → Sync
5. 에뮬레이터를 "독립 창"으로 실행 (Settings>Tools>Emulator 체크 해제) ← 소리!
6. ▶ Run → Demo 탭 → #11 분석
```
