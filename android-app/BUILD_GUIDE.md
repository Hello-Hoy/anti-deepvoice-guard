# Android 앱 빌드 가이드

## 사전 요구사항
- Android Studio Ladybug (2024.2.1) 이상
- JDK 17+
- Android SDK 35 (Compile SDK)
- Android SDK 26+ (Min SDK)

## 빌드 단계

### 1. Android Studio에서 프로젝트 열기
```
File > Open > anti-deepvoice-guard/android-app/
```

### 2. local.properties 설정 (자동 생성됨)
Android Studio가 자동으로 `local.properties`를 생성합니다.
수동으로 필요한 경우:
```
sdk.dir=/Users/<username>/Library/Android/sdk
```

### 3. Assets 확인
`app/src/main/assets/` 디렉토리에 다음 파일이 있어야 합니다:
- `aasist_l.onnx` (660 KB) — AASIST-L deepfake 탐지 모델
- `silero_vad.onnx` (2.3 MB) — Silero VAD 음성 감지 모델

없으면 프로젝트 루트에서:
```bash
# AASIST-L (이미 변환됨)
cp model-server/models/weights/aasist-l.onnx android-app/app/src/main/assets/

# Silero VAD
curl -L https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx \
    -o android-app/app/src/main/assets/silero_vad.onnx
```

### 4. Gradle Sync
Android Studio에서 자동 Sync, 또는:
```bash
./gradlew --refresh-dependencies
```

### 5. Debug APK 빌드
```bash
./gradlew assembleDebug
```
APK 위치: `app/build/outputs/apk/debug/app-debug.apk`

### 6. 실기기 설치 (USB 디버깅 활성화 필요)
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 실기기 테스트 체크리스트

1. [ ] 앱 실행 → 권한 요청 (마이크, 알림) → 허용
2. [ ] START 버튼 → Foreground Service 알림 표시됨
3. [ ] 음성 말하기 → VAD 바 움직임 확인
4. [ ] 5초 이상 말하기 → Detection 결과 카드 업데이트
5. [ ] Deepfake 오디오 스피커 재생 → WARNING/DANGER 알림
6. [ ] STOP 버튼 → 서비스 중지
7. [ ] History 탭 → 탐지 기록 표시
8. [ ] Settings → 엔진 전환, 민감도 조절

## 알려진 제한사항
- 스피커폰 모드에서만 상대방 음성 캡처 가능
- 첫 빌드 시 Gradle 의존성 다운로드로 시간 소요
- Silero VAD의 ONNX 입출력이 버전에 따라 다를 수 있음 (v5 기준)
