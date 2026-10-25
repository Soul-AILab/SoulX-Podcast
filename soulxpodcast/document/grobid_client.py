from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from urllib import error, request


class GrobidError(RuntimeError):
    """Raised when GROBID cannot parse a document."""


class GrobidClient:
    """Small dependency-free client for GROBID's processFulltextDocument API."""

    def __init__(self, base_url: str = "http://localhost:8070", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def process_fulltext(self, pdf_path: str | Path) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {pdf_path}")

        boundary = f"----soulx-{uuid.uuid4().hex}"
        mime = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
        payload = pdf_path.read_bytes()
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="input"; filename="{pdf_path.name}"\r\n'
                    f"Content-Type: {mime}\r\n\r\n"
                ).encode(),
                payload,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        req = request.Request(
            f"{self.base_url}/api/processFulltextDocument",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/xml,text/xml",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GrobidError(f"GROBID returned HTTP {exc.code}: {detail[:500]}") from exc
        except error.URLError as exc:
            raise GrobidError(
                f"Could not reach GROBID at {self.base_url}. Start a GROBID server first."
            ) from exc
