"""File ingestion and export.

Planned capabilities:
- Upload and parse TXT, CSV, PDF, DOCX, XLSX files
- Extract text content for the research agent
- Export session / research results

This module is a placeholder — implementation will follow.
"""

import logging

logger = logging.getLogger(__name__)


def register_ingestion_routes(app):
    """Attach file upload/export endpoints to the FastAPI *app*.

    Will register:
      POST /files/upload
      GET  /files/{file_id}
      GET  /files
    """
    pass  # TODO: implement
