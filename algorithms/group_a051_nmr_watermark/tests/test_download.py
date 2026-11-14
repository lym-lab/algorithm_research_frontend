import hashlib
import io
import zipfile
import pytest
from scripts import download_data as downloader


class Response(io.BytesIO):
    def __init__(self, data, start, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes {start}-{start+len(data)-1}/{total}', 'Content-Length': str(len(data))}


def test_early_eof_resumes_and_verifies_before_extracting(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('fma_medium/test.txt', 'verified music fixture')
    data = stream.getvalue()
    monkeypatch.setitem(downloader.SOURCES, 'fixture', ('https://example.invalid/data', 'sha1', hashlib.sha1(data).hexdigest(), 'fma_medium/'))
    # Emulate a legacy incomplete ZIP and a server ending another response early.
    (tmp_path/'fixture.zip').write_bytes(data[:10])
    offsets = []
    def open_url(request, timeout):
        requested = request.headers['Range']
        if requested == 'bytes=0-0':
            return Response(data[:1], 0, len(data))
        start = int(requested.split('=')[1].split('-')[0])
        offsets.append(start)
        return Response(data[start:start+17], start, len(data))
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(downloader.time, 'sleep', lambda _: None)
    downloader.download('fixture', tmp_path)
    assert offsets[0] == 10 and len(offsets) > 1
    assert (tmp_path/'fixture.zip').read_bytes() == data
    assert (tmp_path/'fma_medium/test.txt').read_text() == 'verified music fixture'
    assert (tmp_path/'fixture.verified.json').exists()


def test_wrong_range_preserves_partial(tmp_path, monkeypatch):
    (tmp_path/'fma.zip.part').write_bytes(b'123')
    def open_url(request, timeout):
        return Response(b'x', 0, 100)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_url)
    with pytest.raises(RuntimeError, match='mismatched range'):
        downloader.download('fma', tmp_path)
    assert (tmp_path/'fma.zip.part').read_bytes() == b'123'
