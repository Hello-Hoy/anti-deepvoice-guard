# TrueVoice Guard Demo Video Storyboard

## Summary

The final video is a 2:30-3:00 concept demo for the practicum competition. It shows two risks and one service response:

1. AI voice impersonation is detected by voice authenticity analysis.
2. Real human voice phishing is detected by STT keyword analysis.
3. Family protection alerts are sent through the MVNO-linked TrueVoice Guard service.

## Scene Timeline

| Time | Scene | Visual | Audio/Text |
|---:|---|---|---|
| 0:00-0:05 | Cold open | Black screen, faint phone vibration, incoming call waveform | Caption: "이제 보이스피싱은 목소리까지 훔칩니다." |
| 0:05-0:15 | Service setup | TrueVoice Guard logo, fast cuts of waveform/STT/alert UI | Narration: "TrueVoice Guard는 딥보이스와 보이스피싱 대화를 동시에 분석하는 AI 음성 보호 서비스입니다." |
| 0:15-0:20 | Ethics caption | Three-panel layout fades in | Caption: "본 영상의 음성은 시연을 위해 사용 허가를 받은 재현 음성입니다." |
| 0:20-1:20 | Scenario 1 | Left: caller/scammer. Center: analysis UI. Right: father answers phone. | Licensed reenactment voice asset. STT and fake probability rise in sync. |
| 1:20-1:30 | Scenario 1 alert | Family phone lock screen, push notification visible for at least 2 seconds | Push: "가족 보호 알림: 가족 - 아버지 의 AI 음성 사칭 의심 통화가 감지되었습니다." |
| 1:30-2:20 | Scenario 2 | Same three-panel layout. Center UI stays identical. | User-recorded real voice phishing reenactment. AI probability stays low; phishing probability rises. |
| 2:20-2:30 | Scenario 2 alert | Family phone lock screen, same notification style | Push: "가족 보호 알림: 가족 - 아버지 의 보이스피싱 키워드가 감지되었습니다." |
| 2:30-2:50 | Business model | MVNO carrier app add-on screen, TrueVoice Guard card in value-added services | Caption: "알뜰폰 통신사 앱에 연동되는 AI 음성 보호 부가서비스" |
| 2:50-3:00 | Closing | TrueVoice Guard logo + three feature chips | Caption: "딥보이스 탐지 + STT 키워드 분석 + 가족 보호 알림" |

## Three-Panel Layout

Use the same composition for Scenario 1 and Scenario 2.

```text
+----------------------+------------------------------+----------------------+
| Caller / Scammer     | TrueVoice Guard Analysis UI  | Father / Victim      |
| dark or neutral tone | bright phone UI, readable    | everyday home scene  |
+----------------------+------------------------------+----------------------+
```

The center panel should be the visual anchor. Keep the side panels slightly darker or less saturated so the audience reads the UI first.

## Center UI Fields

These fields must appear in both scenarios in the same order:

1. `Voice Authenticity`
2. `AI Voice Probability`
3. `STT 실시간 전사`
4. `키워드 하이라이트`
5. `Voice Phishing Probability`
6. `Final Result`

### Scenario 1 Center UI State

| Field | Value |
|---|---|
| Voice Authenticity | `FAKE SUSPECTED` |
| AI Voice Probability | `21% -> 58% -> 92%` |
| STT | `아버지, 지금 급하게 확인해야 해요. 제 계좌가 막혀서 지금 바로 송금이 필요해요.` |
| Highlighted keywords | `급하게`, `계좌`, `지금 바로`, `송금` |
| Voice Phishing Probability | `12% -> 36% -> 61%` |
| Final Result | `DANGER - AI Voice Impersonation Detected` |

### Scenario 2 Center UI State

| Field | Value |
|---|---|
| Voice Authenticity | `REAL` |
| AI Voice Probability | `4%` |
| STT | `서울중앙지검 수사 담당자입니다. 아버님 명의의 계좌가 금융 범죄 수사에 연루된 것으로 확인되었습니다.` |
| Highlighted keywords | `검찰`, `수사`, `계좌`, `이체`, `보호 조치`, `오늘 안에` |
| Voice Phishing Probability | `18% -> 53% -> 87%` |
| Final Result | `WARNING - Voice Phishing Keywords Detected` |

## Push Notification Copy

Use the same format in both scenarios.

### Scenario 1

```text
TrueVoice Guard
가족 보호 알림: 가족 - 아버지 의 AI 음성 사칭 의심 통화가 감지되었습니다.
```

### Scenario 2

```text
TrueVoice Guard
가족 보호 알림: 가족 - 아버지 의 보이스피싱 키워드가 감지되었습니다.
```

## Editing Notes

- Keep all phone notification text on screen for at least 2 seconds.
- Make central UI text large enough to read on a projector.
- Animate probabilities in 2-3 steps instead of smooth continuous motion; the audience should understand the state change instantly.
- Use one UI layout for both scenarios. Only the numbers, highlighted words, and final result should change.
- End with the MVNO business model, not a generic telecom claim.
