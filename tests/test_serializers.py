"""Tests for the serializer module."""

import numpy as np

from dtcc_agent.serializers import serialize


class TestSerializePrimitives:
    def test_int(self):
        result = serialize(42)
        assert result["type"] == "int"
        assert result["value"] == 42

    def test_float(self):
        result = serialize(3.14)
        assert result["type"] == "float"

    def test_string(self):
        result = serialize("hello")
        assert result["type"] == "str"
        assert result["value"] == "hello"

    def test_none(self):
        result = serialize(None)
        assert result["type"] == "NoneType"

    def test_dict(self):
        result = serialize({"a": 1, "b": 2})
        assert result["type"] == "dict"
        assert result["num_keys"] == 2

    def test_bool(self):
        result = serialize(True)
        assert result["type"] == "bool"


class TestSerializeNdarray:
    def test_basic_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = serialize(arr)
        assert result["type"] == "ndarray"
        assert result["shape"] == [3]
        assert result["stats"]["min"] == 1.0
        assert result["stats"]["max"] == 3.0

    def test_empty_array(self):
        arr = np.array([])
        result = serialize(arr)
        assert result["type"] == "ndarray"
        assert result["stats"] is None


class TestSerializeTuple:
    def test_tuple_of_primitives(self):
        result = serialize((1, 2, 3))
        assert result["type"] == "tuple"
        assert result["count"] == 3
        assert len(result["elements"]) == 3

    def test_tuple_of_arrays(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = serialize((a, b))
        assert result["type"] == "tuple"
        assert result["count"] == 2
        assert result["elements"][0]["type"] == "ndarray"


class TestSerializeList:
    def test_generic_list(self):
        result = serialize([1, 2, 3])
        assert result["type"] == "list[int]"
        assert result["count"] == 3


class TestSerializeDolfinxLike:
    """Test serialization of dolfinx-like Function objects."""

    def test_dolfinx_function(self):
        class FakeX:
            array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        class FakeFunc:
            x = FakeX()
        result = serialize(FakeFunc())
        assert result["type"] == "dolfinx.Function"
        assert result["num_dofs"] == 5
        assert result["value_stats"]["mean"] == 3.0


class TestSerializeFallback:
    def test_unknown_type(self):
        class Foo:
            pass
        result = serialize(Foo())
        assert result["type"] == "Foo"
        assert "repr" in result
