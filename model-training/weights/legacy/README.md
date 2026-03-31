# Legacy Weights (Dacon Competition)

Dacon 경진대회(SW중심대학 AI부문) 한국어 데이터셋으로 학습된 가중치.

- `AASIST.pth` — Full AASIST (gat_dims=[64,32]), 1.2MB
- `AASIST-L.pth` — AASIST-L (gat_dims=[24,32]), 416KB

## Known Issue
도메인 시프트 문제: 핸드폰 녹음(m4a 코덱)을 fake로 오판, TTS를 real로 오판.
학습 데이터가 특정 녹음 환경에 편향되어 실세계 일반화 불가.

## Training Details
- Loss: BCEWithLogitsLoss (sigmoid inference)
- Augmentation: RawBoost + DANN domain adaptation
- Data: Dacon 2024 Korean deepfake audio dataset
