"""The web API with the model-free fake service."""
import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tests.fake_service import FakeService
from webapp.server import create_app


def wav_bytes(seconds=1.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def client():
    return TestClient(create_app({"similarity_threshold": 0.25}, FakeService(["knn1", "tts", "mix"])))


def test_systems_grouped():
    j = client().get("/api/systems").json()
    assert [g["name"] for g in j["groups"]] == ["kNN-VC 系", "级联", "混合"]
    assert j["threshold"] == 0.25


def test_anonymize_and_fetch_outputs():
    c = client()
    r = c.post("/api/anonymize", files={"audio": ("a.wav", wav_bytes(), "audio/wav")},
               data={"system": "knn1", "seed": "abc", "p": "0.4", "similarity": "true"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["seed"] == "abc" and j["similarity"] == 0.123 and j["seconds_out"] == 1.25
    wav = c.get(f"/api/result/{j['id']}/anon.wav")
    assert wav.headers["content-type"] == "audio/wav" and sf.info(io.BytesIO(wav.content)).samplerate == 16000
    png = c.get(f"/api/result/{j['id']}/orig.png")
    assert png.content[:4] == b"\x89PNG"


def test_rejects_bad_requests():
    c = client()
    assert c.post("/api/anonymize", files={"audio": ("a.wav", wav_bytes(), "audio/wav")},
                  data={"system": "wc5"}).status_code == 400
    assert c.post("/api/anonymize", files={"audio": ("a.wav", wav_bytes(31), "audio/wav")},
                  data={"system": "knn1"}).status_code == 400
    assert c.get("/api/result/nope/anon.wav").status_code == 404


def test_results_paper_only_and_static_pages():
    c = client()
    rows = c.get("/api/results").json()["rows"]
    assert len(rows) == 12 and rows[1]["paper"]["EER"] == 7.95 and rows[1]["ours"] is None
    for page in ("/", "/results.html", "/method.html", "/style.css", "/demo.js"):
        assert c.get(page).status_code == 200, page
