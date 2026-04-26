import pytest

from soilspec.storage import (
    InMemoryBackend, StorageTier, StorageTierManager,
    embedding_key, map_key, model_key, preprocessed_key, raw_key, temporal_key,
)


def test_get_put_exists_round_trip(storage):
    storage.put(StorageTier.RAW, "p/o1", b"payload")
    assert storage.exists(StorageTier.RAW, "p/o1")
    assert storage.get(StorageTier.RAW, "p/o1") == b"payload"


def test_missing_key_raises(storage):
    with pytest.raises(KeyError):
        storage.get(StorageTier.RAW, "missing")


def test_list_returns_sorted_prefix(storage):
    for k in ("a/2", "a/1", "b/1"):
        storage.put(StorageTier.RAW, k, k)
    assert list(storage.list(StorageTier.RAW, prefix="a/")) == ["a/1", "a/2"]


def test_delete_removes_key(storage):
    storage.put(StorageTier.RAW, "p/o1", "x")
    storage.delete(StorageTier.RAW, "p/o1")
    assert not storage.exists(StorageTier.RAW, "p/o1")


def test_tiers_are_isolated(storage):
    storage.put(StorageTier.RAW, "k", "raw")
    storage.put(StorageTier.MAP, "k", "map")
    assert storage.get(StorageTier.RAW, "k") == "raw"
    assert storage.get(StorageTier.MAP, "k") == "map"


def test_key_helpers_are_distinct():
    assert raw_key("p", "o") != preprocessed_key("a", "t", 0, "m")
    assert embedding_key("t", 0, "m") != preprocessed_key("a", "t", 0, "m")
    assert temporal_key("c") != model_key("c", "v1")
    assert map_key("a", "cap", 1) != map_key("a", "cap", 2)


def test_protocol_satisfied_by_default_backend():
    sm = StorageTierManager(InMemoryBackend())
    sm.put(StorageTier.MODEL, "m/v1", {"weights": []})
    assert sm.exists(StorageTier.MODEL, "m/v1")
