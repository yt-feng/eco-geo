from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def render_pdf_from_html(html_path: Path, pdf_path: Path) -> Path:
    html_path = html_path.resolve()
    pdf_path = pdf_path.resolve()
    binary = shutil.which("wkhtmltopdf")
    if not binary:
        raise RuntimeError(
            "wkhtmltopdf is not installed. Install it locally or run the GitHub Action workflow, which installs it automatically."
        )

    page_format = os.getenv("REPORT_PAGE_FORMAT", "B5").upper()
    if page_format == "A4":
        size_args = ["--page-size", "A4"]
    else:
        # B5-ish consulting booklet canvas. wkhtmltopdf's native B5 varies by build,
        # so use explicit dimensions to keep CSS and PDF renderer aligned.
        size_args = ["--page-width", "176mm", "--page-height", "250mm"]

    command = [
        binary,
        "--enable-local-file-access",
        "--encoding",
        "utf-8",
        "--print-media-type",
        *size_args,
        "--margin-top",
        "0",
        "--margin-right",
        "0",
        "--margin-bottom",
        "0",
        "--margin-left",
        "0",
        str(html_path),
        str(pdf_path),
    ]
    subprocess.run(command, check=True, cwd=str(html_path.parent))
    return pdf_path
