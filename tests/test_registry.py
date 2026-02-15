"""Tests for the function registry."""

import pytest

from dtcc_agent.registry import (
    get_registry,
    get_operation,
    list_operations,
    ParamInfo,
    OperationInfo,
    _extract_params,
    _first_docstring_line,
    _is_object_type,
)

try:
    import dtcc_core
    HAS_DTCC = True
except ImportError:
    HAS_DTCC = False


class TestHelpers:
    def test_is_object_type_pointcloud(self):
        assert _is_object_type("PointCloud") is True

    def test_is_object_type_mesh(self):
        assert _is_object_type("Mesh") is True

    def test_is_object_type_raster(self):
        assert _is_object_type("Raster") is True

    def test_is_object_type_str(self):
        assert _is_object_type("str") is False

    def test_is_object_type_union(self):
        assert _is_object_type("Union[PointCloud, Raster]") is True

    def test_first_docstring_line(self):
        def func():
            """This is a docstring.

            More details here.
            """
        assert _first_docstring_line(func) == "This is a docstring."

    def test_first_docstring_line_none(self):
        def func():
            pass
        assert _first_docstring_line(func) == ""

    def test_extract_params_basic(self):
        def func(x: int, y: float = 3.0):
            pass
        params = _extract_params(func)
        assert len(params) == 2
        assert params[0].name == "x"
        assert params[1].name == "y"
        assert params[1].default == 3.0


class TestParamInfo:
    def test_to_dict_required(self):
        p = ParamInfo(name="x", type_hint="int")
        d = p.to_dict()
        assert d["name"] == "x"
        assert d["required"] is True
        assert "default" not in d

    def test_to_dict_with_default(self):
        p = ParamInfo(name="y", type_hint="float", default=3.0)
        d = p.to_dict()
        assert d["required"] is False
        assert d["default"] == 3.0

    def test_to_dict_object_ref(self):
        p = ParamInfo(name="pc", type_hint="PointCloud", is_object_param=True)
        d = p.to_dict()
        assert d["is_object_ref"] is True


class TestOperationInfo:
    def test_to_dict(self):
        op = OperationInfo(
            name="builder.build_terrain_raster",
            category="builder",
            subcategory="terrain",
            description="Build a terrain raster",
            params=[ParamInfo(name="data", type_hint="PointCloud", is_object_param=True)],
            return_type="Raster",
            tags=["terrain", "raster", "builder"],
        )
        d = op.to_dict()
        assert d["name"] == "builder.build_terrain_raster"
        assert d["category"] == "builder"
        assert len(d["params"]) == 1
        assert d["params"][0]["is_object_ref"] is True


@pytest.mark.skipif(not HAS_DTCC, reason="dtcc_core not installed")
class TestRegistry:
    """Tests that require dtcc-core to be installed."""

    def test_registry_not_empty(self):
        reg = get_registry()
        assert len(reg) > 0, "Registry should have discovered operations"

    def test_builder_functions_registered(self):
        reg = get_registry()
        builder_ops = [k for k in reg if k.startswith("builder.")]
        assert len(builder_ops) > 0, "Should have builder operations"

    def test_datasets_registered(self):
        reg = get_registry()
        ds_ops = [k for k in reg if k.startswith("datasets.")]
        assert len(ds_ops) > 0, "Should have dataset operations"

    def test_io_functions_registered(self):
        reg = get_registry()
        io_ops = [k for k in reg if k.startswith("io.")]
        assert len(io_ops) > 0, "Should have IO operations"

    def test_get_operation_exists(self):
        reg = get_registry()
        first_name = next(iter(reg))
        op = get_operation(first_name)
        assert op.name == first_name

    def test_get_operation_missing(self):
        with pytest.raises(KeyError):
            get_operation("nonexistent.operation")

    def test_list_operations_all(self):
        ops = list_operations()
        assert len(ops) > 0
        assert all("name" in op for op in ops)

    def test_list_operations_filter_category(self):
        ops = list_operations(category="builder")
        assert len(ops) > 0
        assert all(op["category"] == "builder" for op in ops)

    def test_list_operations_search(self):
        ops = list_operations(search="terrain")
        assert len(ops) > 0

    def test_build_terrain_raster_params(self):
        """Verify a key operation has expected parameter structure."""
        op = get_operation("builder.build_terrain_raster")
        assert op.category == "builder"
        param_names = [p.name for p in op.params]
        assert "data" in param_names or "cell_size" in param_names

    def test_dataset_has_bounds_param(self):
        """Datasets should always have a bounds parameter."""
        reg = get_registry()
        ds_ops = [v for k, v in reg.items() if k.startswith("datasets.")]
        for op in ds_ops:
            param_names = [p.name for p in op.params]
            assert "bounds" in param_names, f"Dataset {op.name} missing bounds param"


class TestRegistryNoDtcc:
    """Tests that work without dtcc-core."""

    def test_get_operation_missing(self):
        with pytest.raises(KeyError):
            get_operation("nonexistent.operation")
