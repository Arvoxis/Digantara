"""Render RESPONSE.md to a paginated PDF for submission.

    python make_pdf.py

The assessment caps the written response at four pages, so this exists to make
the page count reproducible and checkable rather than a property of whatever
editor happened to open the Markdown. Uses the `markdown` package plus headless
Chrome, both already present; no LaTeX, no extra dependency.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "RESPONSE.md"
HTML = ROOT / "RESPONSE.html"
PDF = ROOT / "RESPONSE.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 14mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif;
       font-size: 9.3pt; line-height: 1.38; color: #111; margin: 0; }
h1 { font-size: 15pt; margin: 0 0 2mm; border-bottom: 2px solid #333;
     padding-bottom: 1.5mm; }
h2 { font-size: 11.5pt; margin: 4.5mm 0 1.5mm; color: #0b2545;
     border-bottom: 1px solid #ccc; padding-bottom: 0.8mm; }
h3 { font-size: 10pt; margin: 3mm 0 1mm; }
p { margin: 1.4mm 0; text-align: justify; }
ul, ol { margin: 1.4mm 0; padding-left: 5mm; }
li { margin: 0.6mm 0; }
code { font-family: Consolas, monospace; font-size: 8.6pt;
       background: #f2f4f7; padding: 0 1px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0; font-size: 8.4pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1.1mm 1.8mm; text-align: left; }
th { background: #eef1f5; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
hr { border: none; border-top: 1px solid #ccc; margin: 3mm 0; }
strong { color: #000; }
h2, h3 { page-break-after: avoid; }
"""


def find_chrome():
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    for name in ("chrome", "msedge", "chromium"):
        p = shutil.which(name)
        if p:
            return p
    return None


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    body = markdown.markdown(
        SRC.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    HTML.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    print(f"wrote {HTML.name}")

    chrome = find_chrome()
    if not chrome:
        print("Chrome/Edge not found - open RESPONSE.html and print to PDF manually.")
        return 0

    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF}",
            HTML.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    size = PDF.stat().st_size
    pages = PDF.read_bytes().count(b"/Type /Page") - PDF.read_bytes().count(
        b"/Type /Pages"
    )
    print(f"wrote {PDF.name}  ({size / 1024:.0f} KB, ~{pages} pages)")
    if pages > 4:
        print(f"WARNING: {pages} pages exceeds the four-page limit")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
