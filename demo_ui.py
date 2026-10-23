"""
Live-demo page for the three-minute demo. Not part of the graded API.

  uvicorn demo_ui:app --host 0.0.0.0 --port 8000
  open http://localhost:8000/

This imports the real service from app.py unchanged and only adds GET / (a
single static page, no external assets). POST /v1/decide and GET /health are
the same routes the judges hit, so selfcheck.py still works against this port.
"""

import os

from fastapi.responses import FileResponse

from app import app

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "index.html")


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(PAGE, media_type="text/html")
