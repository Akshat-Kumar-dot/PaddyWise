#!/usr/bin/env bash
# Rebuilds report/*.pdf from the executed notebooks and make_report.py.
# Run from the project folder:  bash report/build/build_pdfs.sh
set -e
CH="/c/Program Files/Google/Chrome/Application/chrome.exe"
B="report/build"
UD="$(pwd)/$B/chrome-profile"
export PYTHONIOENCODING=utf-8

print_pdf() {  # $1 = name (html in build folder -> pdf in report folder)
  "$CH" --headless=new --disable-gpu --no-first-run --user-data-dir="$UD" --no-pdf-header-footer \
        --virtual-time-budget=15000 --print-to-pdf="$(cygpath -w "$(pwd)/report/$1.pdf")" \
        "file:///$(cygpath -m "$(pwd)/$B/$1.html")" 2>&1 | tail -1
}

# 1. notebooks -> HTML with print styles -> PDF
for f in Paddy_Yield_Prediction Hybrid_Model_From_Scratch; do
  jupyter nbconvert --to html --embed-images "$f.ipynb" --output-dir "$B" >/dev/null 2>&1
  python - "$B/$f.html" <<'EOF'
import sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
css = """<style>
@page { size: A4; margin: 12mm 10mm 16mm 10mm; }
body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
pre, code, .highlight pre { white-space: pre-wrap !important; word-break: break-word; }
.jp-InputArea-editor, .jp-OutputArea-output { overflow: visible !important; }
img { max-width: 100% !important; page-break-inside: avoid; }
table { page-break-inside: avoid; font-size: 11px; }
</style>"""
open(p, "w", encoding="utf-8").write(s.replace("</head>", css + "</head>", 1))
EOF
  print_pdf "$f"
done

# 2. project report, pass 1 (page numbers unknown)
rm -f "$B/toc_pages.json"
python "$B/make_report.py"
print_pdf Project_Report

# 3. find the page of every heading, then pass 2 with real page numbers
python - "$B" <<'EOF'
import json, sys, fitz
b = sys.argv[1]
heads = json.load(open(f"{b}/toc_heads.json", encoding="utf-8"))
doc = fitz.open("report/Project_Report.pdf")
toc_first = next(i for i in range(doc.page_count) if "Table of Contents" in doc[i].get_text())
last_entry = heads[-1][1][-28:]           # title part of the last heading (its number sits in another column)
toc_last = next(i for i in range(toc_first, doc.page_count) if doc[i].search_for(last_entry))
pages, p = {}, toc_last + 1
abstract = next(i for i in range(doc.page_count) if doc[i].search_for("Abstract"))
pages["Abstract"] = abstract + 1
for level, text in heads:
    key = text[:38]
    while p < doc.page_count and not doc[p].search_for(key):
        p += 1
    if p == doc.page_count:
        raise SystemExit(f"heading not found: {text}")
    pages[text] = p + 1
json.dump(pages, open(f"{b}/toc_pages.json", "w", encoding="utf-8"), ensure_ascii=False)
print("found", len(pages), "headings")
EOF
python "$B/make_report.py"
print_pdf Project_Report

# 4. page numbers in the footer of every PDF (cover page left blank)
python - <<'EOF'
import fitz, os
for name, label in [("Project_Report", "PaddyWise · Project Report · Group 2"),
                    ("Paddy_Yield_Prediction", "Paddy_Yield_Prediction.ipynb"),
                    ("Hybrid_Model_From_Scratch", "Hybrid_Model_From_Scratch.ipynb")]:
    path = f"report/{name}.pdf"
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        if name == "Project_Report" and i == 0:
            continue
        w, h = page.rect.width, page.rect.height
        page.draw_line((40, h - 34), (w - 40, h - 34), color=(0.72, 0.78, 0.85), width=0.6)
        page.insert_text((40, h - 22), label, fontsize=8, color=(0.3, 0.33, 0.38))
        txt = f"Page {i + 1} of {doc.page_count}"
        page.insert_text((w - 40 - fitz.get_text_length(txt, fontsize=8), h - 22), txt, fontsize=8, color=(0.3, 0.33, 0.38))
    doc.save(path + ".tmp"); doc.close(); os.replace(path + ".tmp", path)
print("footers added")
EOF
rm -rf "$UD"
