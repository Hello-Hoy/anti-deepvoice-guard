# TrueVoice Guard Design Image Plan

## Goal

Create Android implementation reference images and presentation scenario images for **TrueVoice Guard**, a real-time AI voice deepfake and voice phishing detection service for MVNO users.

The product is currently a competition demo app, but the service story is B2B2C: provide the APK / SDK-like feature package to MVNO operators so they can add TrueVoice Guard inside their existing carrier apps.

## Naming

- Working name to use in new visuals: **TrueVoice Guard**
- Previous name in code/docs: Anti-DeepVoice Guard / DeepVoice Guard
- Visual copy should emphasize:
  - AI voice protection
  - Real-time call monitoring
  - Deepvoice + voice phishing combined detection
  - On-device privacy

## Design Direction

- Reference source: `ktm_mobile/` screenshots for layout rhythm.
- Button/icon reference: `codex-image/ktm-mobile-buttons/`.
- Style: clean Korean telecom app UI, white background, bold Korean typography, black utility icons, gray inactive states, purple security accent, red danger accent, green safe state.
- Important: use the KTM-style flow as a demo/partner-app concept, not as a final trademarked production screen.

## Android Implementation Reference Images

These images guide the actual Jetpack Compose UI.

Implementation reference images should show **app content only**. Do not include Android/iOS status bars, time, Wi-Fi, battery, device frames, or bottom home indicators. Those areas are system chrome and should not be baked into app UI references.

1. `truevoice-home-v2.png`
   - Main dashboard.
   - Header, security banner, category tabs, large monitoring hero card, quick actions, summary cards.

2. `truevoice-demo-v1.png`
   - Demo scenarios for competition presentation.
   - Six demo cases, waveform/player area, expected result badge.

3. `truevoice-history-v1.png`
   - Detection history list.
   - SAFE/WARNING/DANGER/CRITICAL list rows and empty state.

4. `truevoice-settings-v1.png`
   - On-device inference, STT consent, phishing keyword analysis, auto call monitoring, privacy settings.

5. `truevoice-menu-drawer-v1.png`
   - KTM-style full menu pattern adapted to TrueVoice Guard.
   - Left category rail and right-side feature list.

6. `truevoice-monitoring-active-v1.png`
   - Live monitoring state.
   - STOP action, VAD/STT state, SAFE live threat summary, transcript preview.

7. `truevoice-about-v1.png`
   - Service explanation screen.
   - Feature cards for deepvoice detection, phishing keyword detection, combined threat analysis, and on-device privacy.

## Required State Reference Images

These images guide UI state handling.

1. `truevoice-home-monitoring-active-v1.png`
   - STOP button, live voice activity, STT listening, safe live call state.

2. `truevoice-home-warning-v1.png`
   - WARNING state with phishing keywords highlighted.

3. `truevoice-home-danger-v1.png`
   - DANGER/CRITICAL state with urgent call action.

4. `truevoice-transcription-active-v1.png`
   - Realtime transcript card with detected Korean keywords highlighted.

5. `truevoice-permission-microphone-v1.png`
   - First-run microphone permission education screen.

6. `truevoice-stt-consent-v1.png`
   - On-device STT consent explanation.

## Partner App Presentation Images

These images explain how an MVNO could expose TrueVoice Guard inside an existing carrier app.

1. `partner-ktm-home-banner-truevoice-v1.png`
   - KTM-mobile-style home page.
   - Top hero/banner promotes **TrueVoice Guard** as an AI voice protection add-on.

2. `partner-ktm-menu-open-v1.png`
   - User taps the menu button.
   - Full menu opens with familiar carrier-app structure.
   - Product/security service entry is visible.

3. `partner-ktm-products-selected-v1.png`
   - User taps `상품`.
   - Right-side product menu shows `요금제`, `부가서비스`, `추가서비스`, `휴대폰`.

4. `partner-ktm-value-added-truevoice-v1.png`
   - User taps `부가서비스`.
   - `TrueVoice Guard` appears as a purchasable/activatable security add-on.

5. `partner-ktm-truevoice-detail-v1.png`
   - Optional detail page.
   - Explains AI voice detection, phishing keyword detection, on-device privacy, monthly add-on positioning.

## Data Mapping For Android Implementation

- Monitoring status: `AudioCaptureService.isMonitoring`
- Call banner: `AudioCaptureService.callSession`
- STT banner: `combinedResult.sttStatus`
- Deepfake score: `combinedResult.deepfakeResult.averageFakeScore`
- Phishing score: `combinedResult.phishingScore`
- Combined level: `combinedResult.combinedThreatLevel`
- Keywords: `combinedResult.matchedKeywords`
- Transcript: `AudioCaptureService.transcription`
- Today stats: `AudioCaptureService.stats`
- History rows: `DetectionDatabase.detectionDao().getAll()`
- Settings: `SettingsRepository.settings`

## Reusable Compose Components

- `TrueVoiceTopHeader`
- `TrueVoicePromoBanner`
- `TrueVoiceTabRow`
- `MonitoringHeroCard`
- `QuickActionButton`
- `ThreatSummaryCard`
- `TodayStatsCard`
- `TranscriptCard`
- `DetectionHistoryRow`
- `SettingsToggleRow`
- `PartnerMenuDrawer`

## Design Tokens

- Primary purple: `#7357F6`
- Deep navy: `#090821`
- Safe green: `#20B46B`
- Warning orange: `#F59E0B`
- Danger red: `#EF4444`
- Active black: `#1B1B1B`
- Text primary: `#111111`
- Text secondary: `#666666`
- Surface gray: `#F5F5F5`
- Divider gray: `#E8E8E8`

## Accessibility / Implementation Rules

- Minimum touch target: 48dp.
- Do not communicate threat level by color alone; include text and icon.
- Korean `contentDescription` for tappable icons.
- Keep state cards readable on small Android screens.
- Avoid nested cards; repeated list rows can be card-like, but page sections should stay simple.
- Use Material Icons for utility icons when possible; image assets are references for style and spacing.

## Verification Plan

- Compose Preview for each major state.
- `./gradlew :app:assembleDebug`.
- Emulator or device screenshots for Home, Demo, History, Settings.
- Visual comparison against generated reference images.
