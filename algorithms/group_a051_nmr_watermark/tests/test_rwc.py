import json
from collections import Counter
import numpy as np
from nmr_watermark import data


def test_multifile_piece_has_thirty_segments_without_crossing_parts(tmp_path, monkeypatch):
    for i in range(1, 101):
        for suffix in ('A', 'B', 'C') if i == 58 else ('',):
            (tmp_path/f'RWC_G{i:03d}{suffix}.wav').touch()
    monkeypatch.setattr(data, 'read_audio', lambda _: np.zeros(data.SEGMENT_SAMPLES+100))
    output = tmp_path/'test.json'
    data.prepare_rwc(tmp_path, output)
    original = output.read_bytes()
    rows = json.loads(original)
    assert len(rows) == 3000
    assert set(Counter(r['track_id'] for r in rows).values()) == {30}
    assert all(0 <= r['start'] <= 100 for r in rows)
    assert len({r['path'] for r in rows if r['track_id'] == 'RWC_G058'}) == 3
    data.prepare_rwc(tmp_path, output)
    assert output.read_bytes() == original
