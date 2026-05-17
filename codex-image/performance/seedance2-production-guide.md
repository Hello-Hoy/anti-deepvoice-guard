# TrueVoice Guard Seedance 2.0 Production Guide

This guide replaces the previous video-tool workflow and is written for Seedance 2.0 based production.

Reference guide:

- https://seedance2.ai/ko/guide

## Production Strategy

Use Seedance 2.0 for short image-to-video clips, then assemble the final demo in CapCut or Final Cut Pro.

Best split:

- Seedance 2.0: realistic motion for caller/father scenes, subtle camera life, atmosphere
- CapCut / Final Cut Pro: exact app UI, STT typing, probability count-up, push notifications, captions, audio, final timing

Do not ask Seedance to regenerate text-heavy app UI as video. Korean UI text, percentages, and alert copy must stay exact, so those screens should be animated in the editor.

## Seedance 2.0 Workflow

Use the Image-to-Video workflow:

1. Upload one source image.
2. Paste one motion prompt for that image.
3. Generate a 4-5 second clip.
4. Review hands, phone, face, and object geometry.
5. Regenerate only the failed shot with stronger constraints.

Do not upload multiple story images for the first test. The first goal is to verify that one image can become one stable video clip.

## File Roles

### Generate With Seedance 2.0

Use these as image-to-video source frames:

1. `03_s1_caller_ai_impersonation.png`
2. `04_s1_father_receiving_call.png`
3. `07_s2_caller_real_voice_phishing.png`
4. `08_s2_father_receiving_call.png`

Optional after the four human clips pass:

5. `11_mvno_value_added_service.png`

### Animate In CapCut / Final Cut Pro

Use these as static or motion-graphic layers:

- `01_opening_title.png`
- `02_service_intro.png`
- `05_s1_center_ui_danger.png`
- `06_s1_family_push_ai_impersonation.png`
- `09_s2_center_ui_warning.png`
- `10_s2_family_push_phishing_keyword.png`
- `12_closing_card.png`
- `13_s1_three_panel_composite.png`
- `14_s2_three_panel_composite.png`

## Recommended Seedance 2.0 Settings

Use conservative settings first:

- Mode: Image to Video
- Duration: 4-5 seconds
- Aspect ratio: keep source image / 9:16 for human clips
- Motion strength: low or subtle if available
- Camera motion: minimal
- Resolution: draft first, higher quality after the shot passes
- Audio: off

Why audio off:

- The final voice, alert sound, vibration, and music should be mixed in the editor.
- This prevents unwanted generated speech or mismatched audio timing.

## First Test Order

### Test 1 - Scenario 1 Caller

Use:

- `03_s1_caller_ai_impersonation.png`

Pass criteria:

- Caller face stays hidden.
- Phone stays naturally near the ear.
- Phone screen does not turn toward the camera.
- Laptop remains physically correct.
- Hands do not gain extra fingers.
- Motion is subtle enough for a 3-panel layout.

### Test 2 - Scenario 1 Father

Use:

- `04_s1_father_receiving_call.png`

Pass criteria:

- Father's face does not change identity.
- Phone remains stable.
- Hand remains natural.
- Expression becomes slightly worried but not exaggerated.
- Room background does not warp.

Generate Scenario 2 clips only after these two pass.

## Prompt Pattern

Use short, specific prompts. Seedance should receive motion direction, preservation constraints, and negatives.

```text
Create a realistic 5-second image-to-video clip from this source image.

Preserve:
<identity, composition, lighting, object placement>

Motion:
<small natural movements only>

Style:
<tone and mood>

Avoid:
<specific failures>
```

## Clip Prompts

### Prompt 01 - Scenario 1 Caller

Image:

- `03_s1_caller_ai_impersonation.png`

```text
Create a realistic 5-second image-to-video clip from this source image.

Preserve the anonymous caller, dark room, desk, laptop, smartphone, and low-key lighting. The caller wears a black hoodie and the face must stay hidden in shadow.

The smartphone stays naturally next to the ear. Show only the back or side of the phone. Do not reveal the phone screen. The laptop remains correctly open on the desk, with the keyboard on the bottom and the screen upright.

Motion should be very subtle: faint breathing, slight hand movement, a small head tilt, and a soft laptop screen glow flicker. Keep the camera stable.

Dark documentary reenactment style for a Korean voice phishing prevention demo. Tense, realistic, not horror.

Avoid distorted hands, extra fingers, visible phone screen facing camera, incorrect laptop geometry, screen appearing on the laptop outer cover, warped phone, revealed face, readable private information, logos, text glitches, camera shake, dramatic movement.
```

### Prompt 02 - Scenario 1 Father

Image:

- `04_s1_father_receiving_call.png`

```text
Create a realistic 5-second image-to-video clip from this source image.

Preserve the Korean father, home interior, warm lighting, smartphone, and original composition. He is receiving a suspicious phone call at home.

Motion should be subtle: slight eye movement, small facial expression change, gentle breathing, and his hand tightening a little around the phone. Keep the phone naturally near his ear.

Calm but tense family-protection demo style. Realistic, natural, not melodramatic. Keep the camera stable.

Avoid distorted hands, extra fingers, phone changing shape, face changing identity, exaggerated acting, camera shake, random text, logos, warped background, unrealistic smile, horror style.
```

### Prompt 03 - Scenario 2 Caller

Image:

- `07_s2_caller_real_voice_phishing.png`

```text
Create a realistic 5-second image-to-video clip from this source image.

Preserve the office-like room, caller, smartphone, document, and composition. The caller is reading a suspicious phone script, but the identity should remain partially anonymous.

Motion should be subtle: small hand movement, slight head movement, quiet breathing, and the paper barely moving. Keep the phone and document stable.

Calm but suspicious educational reenactment style for a voice phishing prevention demo. Keep the camera stable.

Avoid distorted hands, extra fingers, phone changing shape, readable official logos, prosecutor or police logos, face changing identity, exaggerated acting, camera shake, random text, warped background, horror style.
```

### Prompt 04 - Scenario 2 Father

Image:

- `08_s2_father_receiving_call.png`

```text
Create a realistic 5-second image-to-video clip from this source image.

Preserve the Korean father, home interior, warm lighting, smartphone, and original composition. He is listening to a suspicious call and gradually looks confused and worried.

Motion should be subtle: slight eye movement, small facial expression change, gentle breathing, and a small hand gesture as he reacts to the call. Keep the camera stable.

Realistic family-protection demo tone. Natural concern, not panic.

Avoid distorted hands, extra fingers, phone changing shape, face changing identity, exaggerated panic, camera shake, random text, logos, warped background, unrealistic smile, horror style.
```

### Optional Prompt 05 - MVNO Service App

Image:

- `11_mvno_value_added_service.png`

```text
Create a subtle 4-second image-to-video clip from this mobile app screen.

Preserve the exact app screen layout as much as possible. Add only a gentle screen parallax or very slow vertical scroll feel. The image should still look like a clean mobile service screen.

Avoid changing Korean text, changing icons, adding random UI elements, warping the phone screen, fake logos, heavy camera motion, and text glitches.
```

If the Korean text changes in this clip, discard the video and use the original still image in CapCut or Final Cut Pro instead.

## Editor Assembly Plan

### Scenario 1

Use Seedance clips:

- Generated video from `03_s1_caller_ai_impersonation.png`
- Generated video from `04_s1_father_receiving_call.png`

Use editor layers:

- `05_s1_center_ui_danger.png`
- `06_s1_family_push_ai_impersonation.png`

Suggested flow:

1. 3-panel layout: caller left, app UI center, father right.
2. Animate STT text typing in the center panel.
3. Count AI Voice Probability up to 92%.
4. Count Voice Phishing Probability up to 61%.
5. Pop in final result: DANGER.
6. Cut to or overlay family push alert.

### Scenario 2

Use Seedance clips:

- Generated video from `07_s2_caller_real_voice_phishing.png`
- Generated video from `08_s2_father_receiving_call.png`

Use editor layers:

- `09_s2_center_ui_warning.png`
- `10_s2_family_push_phishing_keyword.png`

Suggested flow:

1. 3-panel layout: caller left, app UI center, father right.
2. Keep Voice Authenticity as REAL.
3. Keep AI Voice Probability low at 4%.
4. Animate STT transcript and keyword chips.
5. Count Voice Phishing Probability up to 87%.
6. Pop in final result: WARNING.
7. Cut to or overlay family push alert.

## Review Checklist For Generated Clips

Reject and regenerate if:

- Hands are distorted.
- The phone changes direction or shape.
- The caller's hidden face becomes visible.
- The father's face changes identity.
- The laptop has impossible geometry.
- Korean captions become broken.
- The camera moves too much for a 3-panel layout.
- Any real institution or telecom logo appears.
