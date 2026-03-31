# Codex CLI로 Anti-DeepVoice Guard 테스트/검증하기

## 방법 1: Codex CLI에서 직접 코드 리뷰

```bash
cd /Users/hyohee/Documents/Claude_project/anti-deepvoice-guard

# 코드 리뷰 요청
codex "model-training/convert_to_onnx.py 파일을 리뷰해줘. AASIST 모델의 SincConv를 ONNX로 올바르게 변환하고 있는지 확인해줘."

# 서버 코드 리뷰
codex "model-server/ 디렉토리의 FastAPI 서버 코드를 리뷰하고, 보안 이슈나 버그가 있는지 확인해줘."
```

## 방법 2: Codex 서브에이전트로 코드 품질 검증

```bash
# awesome-codex-subagents의 코드 리뷰어 활용
codex "04-quality-security 카테고리의 code-reviewer 에이전트로 anti-deepvoice-guard 프로젝트를 리뷰해줘."
```

## 방법 3: 실제 실행 테스트

```bash
# 1. 모델 변환 테스트 (PyTorch + ONNX Runtime 필요)
cd model-training
pip install -r requirements.txt
python convert_to_onnx.py \
    --model-path ../../deep-fake-audio-detection/code/2_aasist_rawboost/models/weights/AASIST-L.pth \
    --output ../model-server/models/weights/aasist-l.onnx \
    --verify

# 2. 서버 실행 테스트
cd ../model-server
pip install -r requirements.txt
PYTHONPATH=/Users/hyohee/Documents/Claude_project/anti-deepvoice-guard uvicorn model_server.api.main:app --port 8000 &

# 3. API 테스트
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/model/info
# curl -X POST http://localhost:8000/api/v1/detect -F "audio=@test.wav"
```

## 방법 4: Claude Code에서 Codex 서브에이전트 위임

Claude Code 세션에서:
```
awesome-codex-subagents의 test-engineer 또는 code-reviewer 에이전트 설정을 참고하여
anti-deepvoice-guard 프로젝트의 코드를 검증해줘.
```
