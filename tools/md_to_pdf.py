"""Markdown → HTML(with Mermaid) → PDF 변환.

Usage:
    python tools/md_to_pdf.py <input.md> <output.pdf>

Chrome headless로 Mermaid 다이어그램을 렌더링한 뒤 PDF로 인쇄한다.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  body {{
    font-family: -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
    line-height: 1.65;
    color: #222;
    max-width: 900px;
    margin: 0 auto;
    padding: 10px;
    font-size: 11pt;
  }}
  h1 {{ border-bottom: 3px solid #1976d2; padding-bottom: 8px; color: #0d47a1; page-break-before: auto; }}
  h2 {{ border-bottom: 2px solid #bbdefb; padding-bottom: 6px; color: #1565c0; margin-top: 28px; page-break-after: avoid; }}
  h3 {{ color: #1976d2; margin-top: 20px; page-break-after: avoid; }}
  h4 {{ color: #37474f; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; page-break-inside: avoid; }}
  th, td {{ border: 1px solid #cfd8dc; padding: 6px 10px; text-align: left; font-size: 10pt; }}
  th {{ background: #e3f2fd; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-size: 9.5pt;
          font-family: "SF Mono", Menlo, Consolas, monospace; }}
  pre {{ background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 5px; padding: 12px;
         overflow-x: auto; page-break-inside: avoid; }}
  pre code {{ background: transparent; padding: 0; font-size: 9pt; }}
  blockquote {{ border-left: 4px solid #1976d2; margin: 10px 0; padding: 6px 14px;
                background: #e3f2fd; color: #37474f; }}
  .mermaid {{ text-align: center; margin: 18px 0; page-break-inside: avoid; }}
  hr {{ border: none; border-top: 1px solid #cfd8dc; margin: 24px 0; }}
  a {{ color: #1976d2; text-decoration: none; }}
  ul, ol {{ margin: 8px 0 8px 20px; }}
  li {{ margin: 3px 0; }}
</style>
</head>
<body>
{body}
<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'default',
    flowchart: {{ htmlLabels: true, curve: 'basis' }},
    themeVariables: {{ fontFamily: '-apple-system, "Apple SD Gothic Neo", sans-serif' }}
  }});
</script>
</body>
</html>
"""


def convert_mermaid_blocks(md_text: str) -> str:
    """```mermaid ... ``` 블록을 <div class="mermaid"> 로 치환."""
    pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
    return pattern.sub(lambda m: f'<div class="mermaid">\n{m.group(1)}</div>', md_text)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    md_path = Path(sys.argv[1]).resolve()
    pdf_path = Path(sys.argv[2]).resolve()

    md_text = md_path.read_text(encoding="utf-8")
    md_text = convert_mermaid_blocks(md_text)

    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "nl2br"],
    )
    html = HTML_TEMPLATE.format(title=md_path.stem, body=body_html)

    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    print(f"[1/2] HTML 생성: {html_path}")

    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=20000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        f"file://{html_path}",
    ]
    print(f"[2/2] Chrome headless로 PDF 인쇄 중...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"Chrome 오류:\n{result.stderr}")
        return 1

    if not pdf_path.exists():
        print("PDF 파일이 생성되지 않았습니다.")
        return 1

    size_kb = pdf_path.stat().st_size / 1024
    print(f"✅ PDF 생성 완료: {pdf_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
