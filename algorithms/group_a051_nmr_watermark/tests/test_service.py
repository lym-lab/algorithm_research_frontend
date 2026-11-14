import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from nmr_watermark.service import METHOD_ID, create_app


@pytest.fixture
def client(tmp_path):
    # No inference in these malformed-input tests; final-model round trips
    # are independently checked by scripts.validate_service.
    path = tmp_path / 'unused.pt'
    path.write_bytes(b'not loaded by invalid requests')
    registry = tmp_path / 'selected.json'
    entry = {'checkpoint': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    registry.write_text(json.dumps({name: entry for name in ['MSE256', 'NMR256', 'MSE512', 'NMR512']}))
    return TestClient(create_app(registry, tmp_path / 'results'))


@pytest.mark.parametrize('params', ['[]', '{', '{"model":"missing"}',
                                   '{"message":"not binary"}', '{"unknown":1}'])
def test_rejects_invalid_parameters(client, params):
    response = client.post('/process', data={'method_id': METHOD_ID, 'parameters': params},
                           files={'input_file': ('input.wav', b'invalid')})
    assert response.status_code == 400 and response.json()['error']


def test_rejects_undecodable_file_and_allows_cross_origin(client):
    response = client.post('/process', data={'method_id': METHOD_ID},
                           files={'input_file': ('input.wav', b'invalid')},
                           headers={'Origin': 'https://frontend.example'})
    assert response.status_code == 400
    assert response.headers['access-control-allow-origin'] == '*'
    assert '无法解码' in response.json()['error']


def test_health_and_unknown_method(client):
    assert len(client.get('/health').json()['models']) == 4
    response = client.post('/process', data={'method_id': 'wrong'},
                           files={'input_file': ('input.wav', b'invalid')})
    assert response.status_code == 400
