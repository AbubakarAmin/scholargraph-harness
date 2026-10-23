"""Chat control plane — session history separate, harness state shared."""

from pathlib import Path

from fastapi.testclient import TestClient

from web.chat_skills import classify_intent, build_run_overview, help_text, parse_domain_topic_pairs
from web import chat_session


def test_chat_ui_is_primary_landing():
    chat = Path(__file__).resolve().parents[1] / "web" / "static" / "chat" / "index.html"
    assert chat.is_file()
    html = chat.read_text(encoding="utf-8")
    assert "ScholarGraph" in html
    assert "stage-rail" in html
    assert "history-rail" in html
    assert 'id="historyList"' in html
    assert "/dashboard" in html


def test_classify_intent_control_verbs():
    skill, params = classify_intent("Research robust calibration on machine learning")
    assert skill == "start_run"
    blob = f"{params.get('goal') or ''} {params.get('domain') or ''}".lower()
    assert "calibration" in blob or "machine learning" in blob

    assert classify_intent("what's running")[0] == "list_runs"
    assert classify_intent("status of the run")[0] == "status"
    assert classify_intent("approve release abcd1234")[0] == "approve"
    assert classify_intent("summarize this run as markdown")[0] == "generate_doc"
    assert classify_intent("help")[0] == "help"
    assert "dashboard" in help_text().lower()


def test_build_run_overview_scannable():
    ov = build_run_overview(
        {
            "run_id": "deadbeef",
            "status": "running",
            "phase": "engineering",
            "workspace": {
                "selected_topic": {"title": "Calibration gaps"},
                "plan": {"experiments": [{"title": "Ablation A", "status": "planned"}]},
            },
            "events_tail": [{"agent": "engineer", "type": "code_gen"}],
            "release": {"needs_approval": False},
        },
        {"running": True, "run_id": "deadbeef"},
    )
    assert ov["run_id"] == "deadbeef"
    assert ov["status"] == "active"
    assert ov["topic"] == "Calibration gaps"
    assert ov["active_agent"] == "engineer"
    assert ov["plan_experiments"]
    assert any(p["state"] == "active" for p in ov["pipeline"])


def test_chat_session_isolated_from_harness():
    s = chat_session.create_session()
    chat_session.append_message(s["id"], {"role": "user", "content": "hi"})
    chat_session.set_focus_run(s["id"], "cafe0001")
    got = chat_session.get_session(s["id"])
    assert got["focus_run_id"] == "cafe0001"
    assert len(chat_session.list_messages(s["id"])) == 1
    # Creating another session does not share message history
    other = chat_session.create_session()
    assert chat_session.list_messages(other["id"]) == []


def test_chat_api_session_and_harness_endpoints():
    from web.app import app

    client = TestClient(app)
    r = client.post("/api/chat/session", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["session"]["id"]
    assert "harness" in body
    assert "overview" in body["harness"]

    sid = body["session"]["id"]
    g = client.get(f"/api/chat/session/{sid}")
    assert g.status_code == 200
    assert g.json()["messages"] == []

    h = client.get("/api/chat/harness")
    assert h.status_code == 200
    assert "active_runs" in h.json()

    # Help skill does not need LLM / keys
    m = client.post(
        "/api/chat/message",
        json={"session_id": sid, "message": "help"},
    )
    assert m.status_code == 200
    msg = m.json()["message"]
    assert msg["role"] == "assistant"
    assert msg["skill"] == "help"
    assert "harness" in msg["content"].lower() or "run" in msg["content"].lower()
    sessions = client.get("/api/chat/sessions")
    assert sessions.status_code == 200
    listed = next(item for item in sessions.json()["sessions"] if item["id"] == sid)
    assert listed["title"] == "help"
    assert listed["message_count"] == 2


def test_landing_serves_chat_html():
    from web.app import app

    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert b"ScholarGraph" in r.content
    assert b"stage-rail" in r.content or b"stageRail" in r.content

    d = client.get("/dashboard")
    assert d.status_code == 200
    assert b"Control Deck" in d.content or b"Run overview" in d.content


def test_parse_domain_topic_pairs_header_skip():
    text = "domain,topic\nml,calibration\nse,flaky tests\n"
    pairs = parse_domain_topic_pairs(text)
    assert pairs == [
        {"domain": "ml", "topic": "calibration"},
        {"domain": "se", "topic": "flaky tests"},
    ]
