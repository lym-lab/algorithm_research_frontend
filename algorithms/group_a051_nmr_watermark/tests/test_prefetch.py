import pytest
from nmr_watermark.prefetch import prefetch_batches


def test_order_and_empty():
    assert list(prefetch_batches(range(37))) == list(range(37))
    assert list(prefetch_batches([])) == []


def test_reader_error_is_not_swallowed():
    def source():
        yield 1
        raise ValueError('decode failed')
    iterator = prefetch_batches(source())
    assert next(iterator) == 1
    with pytest.raises(ValueError, match='decode failed'):
        next(iterator)
