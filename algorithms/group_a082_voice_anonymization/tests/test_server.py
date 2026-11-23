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


def test_platform_process_contract():
    """POST /process as the course platform sends it: file field, method_id, parameters as a JSON string."""
    c = client()
    r = c.post("/process", files={"audio": ("my clip.wav", wav_bytes(), "audio/wav")},
               data={"method_id": "group_a082_voice_anonymization",
                     "parameters": '{"system": "knn1", "seed": "abc", "mix_p": 0.4}'},
               headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == "*"
    j = r.json()
    assert "相似度 0.123" in j["message"]
    kinds = [a["kind"] for a in j["artifacts"]]
    assert kinds == ["audio", "image", "image", "json"] and all(a["url"].startswith("/api/result/") for a in j["artifacts"])
    for a in j["artifacts"]:
        assert c.get(a["url"]).status_code == 200, a["url"]
    rep = c.get(j["artifacts"][3]["url"]).json()
    assert rep["seed"] == "abc" and rep["system"] == "knn1"


def test_platform_process_errors_are_json_with_cors():
    c = client()
    h = {"Origin": "http://localhost:5173"}
    for data, files in [({"parameters": '{"system": "wc5"}'}, {"audio": ("a.wav", wav_bytes(), "audio/wav")}),
                        ({"parameters": "not json"}, {"audio": ("a.wav", wav_bytes(), "audio/wav")}),
                        ({"parameters": '{"system": "knn1", "mix_p": 2}'}, {"audio": ("a.wav", wav_bytes(), "audio/wav")}),
                        ({"parameters": "{}"}, {"audio": ("a.wav", b"not audio", "audio/wav")}),
                        ({"parameters": "{}"}, None)]:
        r = c.post("/process", data=data, files=files, headers=h)
        assert r.status_code == 400, (data, r.text)
        assert isinstance(r.json()["error"], str) and r.headers["access-control-allow-origin"] == "*"
    pre = c.options("/process", headers={**h, "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 200


def test_upload_size_limit(monkeypatch):
    import webapp.server as srv
    monkeypatch.setattr(srv, "MAX_UPLOAD_MB", 0.01)  # ~10 kB; a 1 s 16-bit clip is 32 kB
    c = client()
    r = c.post("/process", data={"parameters": "{}"}, files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 413 and "MB" in r.json()["error"]


def test_spectrograms_drawn_in_parallel():
    from concurrent.futures import ThreadPoolExecutor
    c = client()
    rid = c.post("/process", data={"parameters": '{"system": "knn1"}'},
                 files={"audio": ("a.wav", wav_bytes(3.0), "audio/wav")}).json()["artifacts"][1]["url"].split("/")[3]
    urls = [f"/api/result/{rid}/{w}.png" for w in ("orig", "anon")] * 8
    with ThreadPoolExecutor(8) as ex:
        pngs = list(ex.map(lambda u: c.get(u).content, urls))
    assert all(p[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs)
    assert len(set(pngs[0::2])) == 1 and len(set(pngs[1::2])) == 1  # identical, not corrupted
