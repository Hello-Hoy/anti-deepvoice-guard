# TrueVoice Guard Demo Video Production Pack

This folder contains the production-ready plan for the practicum competition demo video.

The video is a concept demo, not a live Android call-capture demo. It should present the service as a near-future MVNO add-on where TrueVoice Guard analyzes call audio, STT text, phishing keywords, and family protection alerts.

## Deliverables

- `storyboard.md` - final scene timeline, shot list, UI copy, and edit notes.
- `scenario2-recording-script.md` - sanitized Korean voice-phishing reenactment script for direct phone recording.
- `production-prompts.md` - image/video/UI generation prompts for the needed visual assets.
- `image-generation-shot-prompts.md` - copy-ready still-image prompts by scene, with separate Korean text overlays.
- `central-ui-copy.json` - structured UI states for the editor or motion designer.
- `three-panel-wireframe.html` - browser-viewable reference for the three-panel composition.

## Core Direction

- Runtime: 2:30-3:00.
- Format: one continuous video with two scenarios.
- Layout: both scenarios use the same three-panel composition.
  - Left: caller/scammer side.
  - Center: TrueVoice Guard analysis UI.
  - Right: father/victim side.
- Center UI must keep the same fields in both scenarios:
  - Voice Authenticity
  - AI Voice Probability
  - Real-time STT
  - Keyword Highlight
  - Voice Phishing Probability
  - Final Result

## Required Legal/Ethics Caption

Use this caption once near the start of Scenario 1:

```text
본 영상의 음성은 시연을 위해 사용 허가를 받은 재현 음성입니다.
```

Do not include instructions, tool names, or workflow details for producing unauthorized voice clones in the final video or supporting slides.

## Recommended Image Workflow

1. Use `image-generation-shot-prompts.md` to generate text-free base images in ChatGPT image generation or another image tool.
2. Add exact Korean UI text and notification copy manually in Canva, Figma, Photoshop, CapCut, or an HTML screenshot.
3. Assemble final stills and clips in a video editor using the timing in `storyboard.md`.

## Final Message

```text
TrueVoice Guard
알뜰폰 통신사 앱에 연동되는 AI 음성 보호 부가서비스
딥보이스 탐지 + STT 키워드 분석 + 가족 보호 알림
```
