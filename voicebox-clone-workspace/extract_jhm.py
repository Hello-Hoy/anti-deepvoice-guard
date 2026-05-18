#!/usr/bin/env python3
"""전현무 단독 음성 추출 CLI. 사용법:
  python extract_jhm.py --bootstrap      # 전현무 anchor 자동 발견 + 확인 클립 생성
  python extract_jhm.py --pilot          # 25개 파일 파일럿 (임계 튜닝용)
  python extract_jhm.py --full           # 508개 전체
"""
from jhm_extract.cli import main

if __name__ == "__main__":
    main()
