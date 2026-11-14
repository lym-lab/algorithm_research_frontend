import json
from fastapi.testclient import TestClient
from scripts.listening_server import build_server


def test_collector_saves_session_without_exposing_private_files(tmp_path):
    study, vendor = tmp_path/'study', tmp_path/'vendor'
    (study/'audio').mkdir(parents=True)
    vendor.mkdir()
    (vendor/'index.html').write_text('Test fixture')
    (study/'a051.yaml').write_text('testId: a051')
    (study/'private_mapping.json').write_text('{"C1": "MSE256"}')
    session = {'testId': 'a051', 'trials': [], 'uuid': 'test-fixture'}
    with TestClient(build_server(study, vendor)) as client:
        assert client.get('/configs/a051.yaml').text == 'testId: a051'
        assert client.post('/service/write.php', content='sessionJSON='+json.dumps(session)).status_code == 200
        saved = list((study/'responses').glob('*.json'))
        assert len(saved) == 1 and json.loads(saved[0].read_text()) == session
        for path in ['/private_mapping.json', '/study/private_mapping.json', '/responses/'+saved[0].name]:
            assert client.get(path).status_code == 404
        for body in ['bad', 'sessionJSON={', 'sessionJSON=[]', 'sessionJSON={"testId":"other","trials":[]}']:
            assert client.post('/service/write.php', content=body).status_code == 400
        assert len(list((study/'responses').glob('*.json'))) == 1
