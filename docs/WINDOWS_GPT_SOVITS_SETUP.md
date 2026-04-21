# Windows 11 + RTX 4080 Super — GPT-SoVITS Phase B 셋업 가이드

이 문서는 Mac(M4 Pro)에서 녹음/전처리 완료한 voice cloning 데이터를 Windows 데스크탑(RTX 4080 Super)으로 옮겨 **Phase B (GPT-SoVITS) 파인튜닝**을 진행하는 절차입니다.

## 전제

- Windows 11
- NVIDIA RTX 4080 Super (CUDA 12.x 호환)
- 외장 디스크에 Mac에서 복사해온 녹음 데이터:
  - `raw_hyohee/` (48kHz 24bit WAV 150개, 총 91MB)
  - `user-hyohee_preprocessed/` (전처리된 24kHz/16kHz + manifest, 참고용)
  - `phase0_ref_*.wav` (10초 레퍼런스)

Mac에서의 진행 상황:
- XTTS v2 150 epoch 파인튜닝 시도 → 과적합 + MPS 한계로 품질 부족 판단
- 이 프로젝트의 최고 품질 경로는 **GPT-SoVITS (CUDA 필수)** 로 결론

---

## 1단계. 시스템 도구 설치

### 1.1 NVIDIA 드라이버 / CUDA Toolkit
```powershell
nvidia-smi
```
- 버전 출력되면 OK. 없으면 https://www.nvidia.com/Download/index.aspx 에서 RTX 4080 Super 드라이버 설치
- CUDA Toolkit 12.1~12.4 권장: https://developer.nvidia.com/cuda-downloads

### 1.2 Python 3.11
- https://www.python.org/downloads/release/python-3119/ → Windows installer (64-bit)
- 설치 시 "Add Python to PATH" 체크

### 1.3 Git for Windows
- https://git-scm.com/download/win

### 1.4 Node.js 18+ (Claude Code CLI 전제)
- https://nodejs.org/ → LTS 다운로드

### 1.5 Claude Code CLI
PowerShell:
```powershell
npm install -g @anthropic-ai/claude-code
claude login  # 브라우저 열려서 Anthropic 계정 로그인
```

### 1.6 ffmpeg (오디오 변환)
- https://www.gyan.dev/ffmpeg/builds/ → release full 다운로드
- 압축 해제 후 `bin/` 폴더를 PATH에 추가 (예: `C:\ffmpeg\bin`)
- 확인: `ffmpeg -version`

---

## 2단계. 프로젝트 clone

```powershell
cd C:\
mkdir Projects
cd Projects
git clone https://github.com/Hello-Hoy/anti-deepvoice-guard.git
cd anti-deepvoice-guard
```

## 3단계. 녹음 데이터 외장 디스크 → 프로젝트 복사

PowerShell (외장 디스크가 `E:` 라 가정):
```powershell
# 원본 녹음 150 WAV (GPT-SoVITS 학습용)
mkdir -Force model-training\voice-clone\data\raw\hyohee
Copy-Item -Recurse -Path "E:\voice-clone-data\raw_hyohee\session_*" -Destination "model-training\voice-clone\data\raw\hyohee\"

# 전처리 완료본 (참고용, GPT-SoVITS는 자체 전처리함)
mkdir -Force model-training\voice-clone\data\user-hyohee
Copy-Item -Recurse -Path "E:\voice-clone-data\user-hyohee_preprocessed\*" -Destination "model-training\voice-clone\data\user-hyohee\"

# Phase 0 ref
mkdir -Force test-samples\voice-clone-raw
Copy-Item "E:\voice-clone-data\phase0_ref_*.wav" "test-samples\voice-clone-raw\"
```

---

## 4단계. Python venv + 기본 의존성

```powershell
py -3.11 -m venv .venv-cloning
.venv-cloning\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

프로젝트 코드 의존성 (평가/stress 테스트 등에 필요, 선택):
```powershell
pip install -r model-training\requirements-cloning.txt
```

transformers 다운그레이드 (Coqui TTS 호환):
```powershell
pip install "transformers==4.40.2"
```

PyTorch CUDA 12.1 wheel:
```powershell
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

검증:
```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```
`cuda: True`, device에 RTX 4080 Super 나와야 OK.

---

## 5단계. GPT-SoVITS 설치

프로젝트 외부 경로에 clone (별도 venv):
```powershell
cd C:\Projects
git clone https://github.com/RVC-Boss/GPT-SoVITS
cd GPT-SoVITS
```

안정 태그 체크아웃 (권장):
```powershell
git fetch --tags
git tag --list | Select-Object -First 10
# 최신 stable 태그로 체크아웃 (예시)
# git checkout v2-stable
```

Python venv 별도 구성 (GPT-SoVITS 내부):
```powershell
py -3.11 -m venv .venv-gpt-sovits
.venv-gpt-sovits\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

사전학습 모델 다운로드 (GPT-SoVITS README 참고):
```powershell
# HuggingFace 또는 공식 다운로드 스크립트
# 일반적으로 tools/pretrained/ 하위에 배치
# huggingface-cli download lj1995/GPT-SoVITS --local-dir GPT_SoVITS/pretrained_models
```

---

## 6단계. 녹음 데이터를 GPT-SoVITS 포맷으로

GPT-SoVITS 공식 WebUI 또는 CLI 사용. 우리 프로젝트에 이미 **engines/gpt_sovits.py adapter** 가 있어서 이걸 활용하는 게 가장 간단합니다.

anti-deepvoice-guard 프로젝트 폴더로 돌아가서:
```powershell
cd C:\Projects\anti-deepvoice-guard

# 환경변수 설정 (이 세션만)
$env:GPT_SOVITS_DIR = "C:\Projects\GPT-SoVITS"
$env:GPT_SOVITS_VENV = "C:\Projects\GPT-SoVITS\.venv-gpt-sovits\Scripts\python.exe"

# health_check 확인
.venv-cloning\Scripts\Activate.ps1
python -c "from model_training.voice_clone.engines import load_engine; e = load_engine('gpt_sovits'); ok, msg = e.health_check(); print(ok); print(msg)"
```

health_check가 통과하면 prepare_data → train 실행:
```powershell
python -c "
from pathlib import Path
from model_training.voice_clone.engines import load_engine
eng = load_engine('gpt_sovits')
prepared = eng.prepare_data(
    manifest_path=Path('model-training/voice-clone/data/user-hyohee/manifest.json'),
    out_dir=Path('model-training/voice-clone/data/user-hyohee/gpt_sovits_prepared'),
)
print('prepared:', prepared)
"
```

학습:
```powershell
python -c "
from pathlib import Path
from model_training.voice_clone.engines import load_engine
eng = load_engine('gpt_sovits')
ckpt = eng.train(
    data_dir=Path('model-training/voice-clone/data/user-hyohee/gpt_sovits_prepared'),
    ckpt_out=Path('model-training/weights/voice-clone/user-hyohee/gpt_sovits'),
    s2_epochs=8,
    s1_epochs=15,
)
print('ckpt:', ckpt)
"
```

예상 시간: RTX 4080 Super에서 **S2 학습 4~6시간 + S1 학습 2~3시간 = 6~9시간**.

---

## 7단계. 합성 + 평가

학습 완료 후 추론:
```powershell
python -c "
from pathlib import Path
from model_training.voice_clone.engines import load_engine
eng = load_engine('gpt_sovits')
eng.synthesize(
    text='여보세요, 저 지금 급해서요. 계좌번호 한 번만 다시 불러주세요.',
    ckpt=Path('model-training/weights/voice-clone/user-hyohee/gpt_sovits'),
    ref_wav=Path('test-samples/voice-clone-raw/phase0_ref_20260420_205403.wav'),
    language='ko',
    out_path=Path('test-samples/Demo/phase_b_clone.wav'),
)
"
```

품질 평가 (AASIST + 화자 유사도):
```powershell
python model-training\voice-clone\eval_quality.py `
  --wav test-samples\Demo\phase_b_clone.wav `
  --text "여보세요, 저 지금 급해서요. 계좌번호 한 번만 다시 불러주세요." `
  --ref-wav test-samples\voice-clone-raw\phase0_ref_20260420_205403.wav `
  --aasist-onnx model-training\weights\finetuned\aasist_embedded.onnx `
  --metrics wer,utmosv2,dnsmos,spk_sim,aasist
```

탐지기 stress test:
```powershell
python model-training\voice-clone\detector_stress.py `
  --clone-dir test-samples\Demo `
  --real-dir model-training\voice-clone\data\raw\hyohee\session_20260420_211638 `
  --aasist-onnx model-training\weights\finetuned\aasist_embedded.onnx `
  --out reports\voice-clone\stress_phase_b
```

---

## 8단계. 체크포인트를 Mac으로 다시 가져오기 (선택)

외장 디스크 또는 rsync:
```powershell
# 외장 디스크에 복사
Copy-Item -Recurse "model-training\weights\voice-clone\user-hyohee\gpt_sovits" "E:\voice-clone-data\gpt_sovits_ckpt"
```

Mac에서:
```bash
mkdir -p model-training/weights/voice-clone/user-hyohee/
cp -R /Volumes/MyDrive/voice-clone-data/gpt_sovits_ckpt model-training/weights/voice-clone/user-hyohee/gpt_sovits
```

---

## 트러블슈팅

### CUDA not available
- `nvidia-smi` 작동 확인
- torch 설치 시 `--index-url https://download.pytorch.org/whl/cu121` 누락 여부 확인

### GPT-SoVITS requirements.txt 설치 실패
- Visual C++ Build Tools 필요할 수 있음: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- 또는 특정 패키지만 개별 설치 (`pip install <pkg> --no-build-isolation`)

### PowerShell 실행 정책 오류
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 한글 경로 이슈
- 프로젝트 경로에 한글/공백 없는 것 권장 (`C:\Projects\anti-deepvoice-guard` OK)

---

## Claude Code 이어서 사용

데스크탑의 Claude Code에서 이 대화 컨텍스트를 이어가려면:
1. `C:\Projects\anti-deepvoice-guard` 에서 `claude` 실행
2. 새 세션이 시작되면 먼저 이 파일 읽으라고 지시:
   ```
   /Users/hyohee/.claude/plans/deepresearch-deepfake-curried-snowflake.md 와 
   docs/WINDOWS_GPT_SOVITS_SETUP.md 를 먼저 읽고 현재까지 진행 상황 파악해줘.
   ```
   (Windows에선 `C:\Projects\anti-deepvoice-guard\docs\WINDOWS_GPT_SOVITS_SETUP.md` 로 경로 적어주세요)
3. 새 Claude가 전체 맥락을 파악한 후 Phase B 학습 실행부터 이어서 진행

---

## 이 문서의 기준 커밋
- Mac에서의 진행 상태가 반영된 마지막 commit: `55490be` (또는 최신 master)
- 데스크탑에서 `git pull` 먼저 실행 권장
