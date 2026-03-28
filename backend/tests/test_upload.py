"""
Tests — Upload endpoint
"""

import io
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _create_dummy_png() -> bytes:
    """Create a minimal valid 1x1 PNG in memory."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("L", (64, 64), color=128)
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.anyio
async def test_upload_valid_png():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        png_bytes = _create_dummy_png()
        response = await client.post(
            "/api/v1/upload/",
            files={"file": ("test_scan.png", png_bytes, "image/png")},
        )
    assert response.status_code == 201
    data = response.json()
    assert "scan_id" in data
    assert data["filename"] == "test_scan.png"


@pytest.mark.anyio
async def test_upload_invalid_extension():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/upload/",
            files={"file": ("readme.txt", b"hello world", "text/plain")},
        )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]
