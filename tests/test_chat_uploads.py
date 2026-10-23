import asyncio
from io import BytesIO

from core.config import config
from core.packs import load_pack
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from web.chat_api import (
    _is_domain_topic_list_upload,
    _read_upload_limited,
    _upload_as_pack,
    _has_domain_topic_header,
)
from web.chat_skills import parse_domain_topic_pairs


def test_chat_upload_registers_pack_with_safe_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "output_dir", str(tmp_path))

    reply = asyncio.run(
        _upload_as_pack(
            "session",
            "../../outside.csv",
            b"participant,score\np1,4\np2,5\n",
            "text/csv",
        )
    )

    artifact = reply["artifacts"][0]
    manifest = load_pack(artifact["pack_id"], str(tmp_path))
    assert manifest is not None
    table_path = manifest["response_table"]["manifest"]["path"]
    assert table_path.startswith(str(tmp_path))
    assert table_path.endswith(".csv")
    assert "outside.csv" == manifest["response_table"]["manifest"]["name"]


def test_chat_upload_rejects_invalid_pack_and_cleans_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "output_dir", str(tmp_path))

    reply = asyncio.run(
        _upload_as_pack(
            "session",
            "responses.csv",
            b"not,a,valid\n",
            "text/csv",
        )
    )

    assert not reply["artifacts"]
    assert "Could not register" in reply["content"]
    pack_root = tmp_path / "packs"
    assert not list(pack_root.glob("*/**/*"))


def test_csv_upload_routing_distinguishes_topic_lists_from_data_tables():
    topic_list = "domain,topic\nmachine learning,robust calibration\n"
    pairs = parse_domain_topic_pairs(topic_list)
    assert pairs == [{"domain": "machine learning", "topic": "robust calibration"}]
    assert _has_domain_topic_header(topic_list)
    assert _is_domain_topic_list_upload(topic_list, "Uploaded file", pairs)

    data_table = "participant,score\np1,4\np2,5\n"
    data_pairs = parse_domain_topic_pairs(data_table)
    assert data_pairs == [
        {"domain": "participant", "topic": "score"},
        {"domain": "p1", "topic": "4"},
        {"domain": "p2", "topic": "5"},
    ]
    assert not _is_domain_topic_list_upload(data_table, "Uploaded file", data_pairs)


def test_chat_upload_rejects_bytes_over_limit():
    upload = UploadFile(file=BytesIO(b"more than limit"), filename="data.csv")

    try:
        asyncio.run(_read_upload_limited(upload, 4))
    except HTTPException as exc:
        assert exc.status_code == 413
    else:
        raise AssertionError("oversized upload should be rejected")


def test_interpret_uploads_sends_images_as_gemini_media(monkeypatch):
    import web.chat_api as chat_api

    captured = {}

    def fake_call_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return "The chart trends upward."

    monkeypatch.setattr("core.llm.call_llm", fake_call_llm)
    reply = asyncio.run(
        chat_api._interpret_uploads(
            "Describe the chart",
            [{
                "name": "chart.png",
                "raw": b"image-bytes",
                "mime_type": "image/png",
                "text": "",
            }],
        )
    )

    assert captured["media_parts"] == [{"data": b"image-bytes", "mime_type": "image/png"}]
    assert "Describe the chart" in captured["prompt"]
    assert reply["content"] == "The chart trends upward."


def test_gemini_chat_builds_inline_media_parts(monkeypatch):
    from types import SimpleNamespace
    import core.llm as llm_mod

    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            part = SimpleNamespace(text="image understood")
            candidate = SimpleNamespace(
                content=SimpleNamespace(parts=[part]),
                finish_reason="STOP",
            )
            return SimpleNamespace(candidates=[candidate])

    models = FakeModels()
    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    client._gemini = SimpleNamespace(models=models)
    monkeypatch.setattr(
        type(llm_mod.config),
        "is_gemma_4_model",
        lambda self, model=None: False,
    )

    result = client._chat_gemini(
        "Describe the attachment",
        0.2,
        "gemma-4-26b-a4b-it",
        64,
        None,
        media_parts=[{"data": b"image-bytes", "mime_type": "image/png"}],
    )

    request_parts = models.calls[0]["contents"]
    assert result == "image understood"
    assert request_parts[0] == "Describe the attachment"
    assert request_parts[1].inline_data.data == b"image-bytes"
    assert request_parts[1].inline_data.mime_type == "image/png"


def test_chat_upload_keeps_prompt_with_queued_file(monkeypatch):
    import web.chat_api as chat_api
    from web.app import app
    from fastapi.testclient import TestClient

    captured = {}

    async def fake_interpret(note, files):
        captured["note"] = note
        captured["files"] = files
        return chat_api._assistant_message("Analyzed your attachment.", skill="upload_pack")

    monkeypatch.setattr(chat_api, "_interpret_uploads", fake_interpret)
    monkeypatch.setattr(chat_api, "_public_harness", lambda run_id: {})

    response = TestClient(app).post(
        "/api/chat/upload",
        data={"message": "What does this image show?"},
        files=[("files", ("figure.png", b"png-data", "image/png"))],
    )

    assert response.status_code == 200
    assert captured["note"] == "What does this image show?"
    assert captured["files"][0]["name"] == "figure.png"
    assert captured["files"][0]["raw"] == b"png-data"
    assert response.json()["message"]["content"] == "Analyzed your attachment."


def test_dashboard_pack_upload_limits_read_before_persisting(tmp_path, monkeypatch):
    import core.packs as packs
    from web.app import upload_pack

    monkeypatch.setattr(config, "output_dir", str(tmp_path))
    monkeypatch.setattr(packs, "RESPONSE_TABLE_MAX_BYTES", 4)

    class OversizedUpload:
        filename = "responses.csv"
        size = None

        async def read(self, size):
            assert size == 5
            return b"12345"

    class FakeRequest:
        async def form(self):
            return {"response_table": OversizedUpload()}

    response = asyncio.run(upload_pack(FakeRequest()))
    assert response.status_code == 413
    assert not list((tmp_path / "packs" / "_staging").glob("*"))
