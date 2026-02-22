"""Function registry: discovers and catalogs all dtcc-core operations.

Scans dtcc-core modules at import time to build a searchable catalog.
Each registered operation stores its name, category, description,
parameter info (including which params accept dtcc-core objects), and tags.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, get_type_hints

logger = logging.getLogger(__name__)

# dtcc-core model types whose parameters should be resolved from the ObjectStore.
_OBJECT_TYPE_NAMES: set[str] = {
    "PointCloud", "Mesh", "VolumeMesh", "Raster",
    "City", "Building", "Terrain", "Tree",
    "Surface", "MultiSurface", "RoadNetwork",
    "Bounds", "Object",
}


@dataclass
class ParamInfo:
    name: str
    type_hint: str = ""
    default: Any = inspect.Parameter.empty
    is_object_param: bool = False

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type_hint,
            "required": self.default is inspect.Parameter.empty,
        }
        if self.default is not inspect.Parameter.empty:
            # Make default JSON-serializable
            try:
                if self.default is None or isinstance(self.default, (int, float, str, bool, list)):
                    d["default"] = self.default
                else:
                    d["default"] = repr(self.default)
            except Exception:
                d["default"] = repr(self.default)
        if self.is_object_param:
            d["is_object_ref"] = True
        return d


@dataclass
class OperationInfo:
    name: str  # e.g. "builder.build_terrain_raster"
    category: str  # e.g. "builder"
    subcategory: str = ""  # e.g. "terrain"
    description: str = ""
    params: list[ParamInfo] = field(default_factory=list)
    return_type: str = ""
    tags: list[str] = field(default_factory=list)
    _callable: Any = None  # the actual function

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "subcategory": self.subcategory,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
            "return_type": self.return_type,
            "tags": self.tags,
        }


def _is_object_type(type_str: str) -> bool:
    """Check if a type annotation string refers to a dtcc-core model type."""
    for name in _OBJECT_TYPE_NAMES:
        if name in type_str:
            return True
    return False


def _extract_params(func: Any) -> list[ParamInfo]:
    """Extract parameter info from a function's signature and type hints."""
    params = []
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return params

    # Try to get type hints, fall back to annotation strings
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    for pname, p in sig.parameters.items():
        if pname in ("self", "cls"):
            continue

        # Get type string — use str(hint) to preserve Union inner types
        # (getattr(hint, "__name__") returns "Union" which loses PointCloud/Raster/etc.)
        if pname in hints:
            hint = hints[pname]
            type_str = str(hint)
        elif p.annotation is not inspect.Parameter.empty:
            type_str = str(p.annotation)
        else:
            type_str = ""

        default = p.default
        is_obj = _is_object_type(type_str)

        params.append(ParamInfo(
            name=pname,
            type_hint=type_str,
            default=default,
            is_object_param=is_obj,
        ))
    return params


def _get_return_type(func: Any) -> str:
    """Extract return type annotation as a string."""
    try:
        hints = get_type_hints(func)
        ret = hints.get("return")
        if ret is not None:
            return getattr(ret, "__name__", str(ret))
    except Exception:
        pass
    return ""


def _first_docstring_line(func: Any) -> str:
    """Extract the first non-empty line of a docstring."""
    doc = getattr(func, "__doc__", None)
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _make_tags(name: str, category: str, description: str) -> list[str]:
    """Generate search tags from name, category, and description."""
    tags = set()
    # Split name parts
    for part in name.replace(".", "_").split("_"):
        if len(part) > 2:
            tags.add(part.lower())
    tags.add(category.lower())
    # Key words from description
    for word in description.lower().split():
        if len(word) > 3 and word.isalpha():
            tags.add(word)
    return sorted(tags)


# ── Registration helpers ─────────────────────────────────────────────

def _register_functions(
    registry: dict[str, OperationInfo],
    module: Any,
    category: str,
    subcategory: str = "",
    names: list[str] | None = None,
    prefix: str = "",
) -> None:
    """Register functions from a module into the registry."""
    if names is None:
        names = getattr(module, "__all__", None)
        if names is None:
            names = [n for n in dir(module)
                     if not n.startswith("_") and callable(getattr(module, n, None))]

    for fname in names:
        func = getattr(module, fname, None)
        if func is None or not callable(func):
            continue
        # Skip classes, only register functions
        if isinstance(func, type):
            continue

        op_name = f"{prefix}{fname}" if prefix else f"{category}.{fname}"
        if op_name in registry:
            continue

        desc = _first_docstring_line(func)
        params = _extract_params(func)
        ret_type = _get_return_type(func)
        tags = _make_tags(op_name, category, desc)

        registry[op_name] = OperationInfo(
            name=op_name,
            category=category,
            subcategory=subcategory,
            description=desc,
            params=params,
            return_type=ret_type,
            tags=tags,
            _callable=func,
        )


def _register_datasets(registry: dict[str, OperationInfo]) -> None:
    """Register dtcc-core datasets as operations."""
    try:
        from dtcc_core.datasets.registry import list_datasets
    except ImportError:
        return

    datasets = list_datasets()
    for ds_name, ds_instance in datasets.items():
        op_name = f"datasets.{ds_name}"
        desc = getattr(ds_instance, "description", "") or f"Dataset: {ds_name}"
        if isinstance(desc, str):
            desc = desc.strip().split("\n")[0]

        # Extract params from ArgsModel schema
        params = [ParamInfo(name="bounds", type_hint="list[float]",
                           default=inspect.Parameter.empty, is_object_param=False)]
        try:
            schema = ds_instance.show_options()
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            for pname, pinfo in props.items():
                if pname == "bounds":
                    continue
                ptype = pinfo.get("type", pinfo.get("anyOf", ""))
                pdefault = pinfo.get("default", inspect.Parameter.empty)
                params.append(ParamInfo(
                    name=pname,
                    type_hint=str(ptype),
                    default=pdefault,
                    is_object_param=False,
                ))
        except Exception:
            pass

        tags = _make_tags(op_name, "datasets", desc)

        registry[op_name] = OperationInfo(
            name=op_name,
            category="datasets",
            subcategory=ds_name,
            description=desc,
            params=params,
            return_type="",
            tags=tags,
            _callable=ds_instance,
        )


# ── Build the global registry ─────────────────────────────────────────

_REGISTRY: dict[str, OperationInfo] | None = None


def _build_registry() -> dict[str, OperationInfo]:
    """Scan dtcc-core modules and build the operation catalog."""
    registry: dict[str, OperationInfo] = {}

    # 1. Builder top-level functions
    try:
        import dtcc_core.builder as builder_mod
        _register_functions(registry, builder_mod, "builder", "general",
                          names=builder_mod.__all__)
    except Exception as e:
        logger.warning(f"Failed to register builder functions: {e}")

    # 2. Pointcloud filter functions
    try:
        from dtcc_core.builder.pointcloud import filter as pc_filter
        _register_functions(registry, pc_filter, "builder", "pointcloud_filter",
                          prefix="builder.pc_filter.")
    except Exception as e:
        logger.warning(f"Failed to register pointcloud filter functions: {e}")

    # 3. Pointcloud convert
    try:
        from dtcc_core.builder.pointcloud import convert as pc_convert
        _register_functions(registry, pc_convert, "builder", "pointcloud_convert",
                          prefix="builder.pc_convert.")
    except Exception as e:
        logger.warning(f"Failed to register pointcloud convert functions: {e}")

    # 4. Raster operations
    try:
        from dtcc_core.builder.raster import analyse as raster_analyse
        _register_functions(registry, raster_analyse, "builder", "raster_analysis",
                          prefix="builder.raster.")
    except Exception as e:
        logger.warning(f"Failed to register raster analyse functions: {e}")

    try:
        from dtcc_core.builder.raster import filter as raster_filter
        _register_functions(registry, raster_filter, "builder", "raster_filter",
                          prefix="builder.raster.")
    except Exception as e:
        logger.warning(f"Failed to register raster filter functions: {e}")

    try:
        from dtcc_core.builder.raster import stats as raster_stats
        _register_functions(registry, raster_stats, "builder", "raster_stats",
                          prefix="builder.raster.")
    except Exception as e:
        logger.warning(f"Failed to register raster stats functions: {e}")

    try:
        from dtcc_core.builder.raster import interpolation as raster_interp
        _register_functions(registry, raster_interp, "builder", "raster_interpolation",
                          prefix="builder.raster.")
    except Exception as e:
        logger.warning(f"Failed to register raster interpolation functions: {e}")

    # 5. Meshing functions
    try:
        from dtcc_core.builder import meshing as meshing_mod
        meshing_names = [
            "mesh_multisurface", "mesh_surface", "mesh_multisurfaces",
            "merge_meshes", "disjoint_meshes", "snap_vertices", "merge",
            "tile_surface_mesh", "extrude_surface_to_solid",
            "create_printable_surface_mesh",
        ]
        _register_functions(registry, meshing_mod, "builder", "meshing",
                          names=meshing_names, prefix="builder.meshing.")
    except Exception as e:
        logger.warning(f"Failed to register meshing functions: {e}")

    # 6. IO functions
    try:
        import dtcc_core.io as io_mod
        _register_functions(registry, io_mod, "io", "general",
                          names=io_mod.__all__)
    except Exception as e:
        logger.warning(f"Failed to register IO functions: {e}")

    # 7. Datasets
    try:
        # Import dtcc_sim datasets to trigger registration first
        try:
            import dtcc_sim.datasets  # noqa: F401
        except ImportError:
            pass
        _register_datasets(registry)
    except Exception as e:
        logger.warning(f"Failed to register datasets: {e}")

    # 8. Reproject functions
    try:
        from dtcc_core.reproject import reproject as reproject_mod
        reproject_names = [
            "reproject_array", "reproject_surface", "reproject_mesh",
            "reproject_multisurface", "reproject_object", "reproject_pointcloud",
        ]
        _register_functions(registry, reproject_mod, "reproject", "general",
                          names=reproject_names)
    except Exception as e:
        logger.warning(f"Failed to register reproject functions: {e}")

    logger.info(f"Registry built with {len(registry)} operations")
    return registry


def get_registry() -> dict[str, OperationInfo]:
    """Return the global operation registry, building it on first call."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_operation(name: str) -> OperationInfo:
    """Look up a single operation by name. Raises KeyError if not found."""
    reg = get_registry()
    if name not in reg:
        raise KeyError(f"Operation '{name}' not found. Use list_operations() to see available operations.")
    return reg[name]


def list_operations(
    category: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """List operations, optionally filtered by category or search term.

    Parameters
    ----------
    category : str, optional
        Filter to operations in this category (e.g. "builder", "io", "datasets").
    search : str, optional
        Free-text search — matches against name, description, and tags.

    Returns
    -------
    list of dicts with name, category, description for each matching operation.
    """
    reg = get_registry()
    results = []

    search_lower = search.lower() if search else None

    for op in reg.values():
        if category and op.category != category:
            continue
        if search_lower:
            searchable = f"{op.name} {op.description} {' '.join(op.tags)}".lower()
            if search_lower not in searchable:
                continue
        results.append({
            "name": op.name,
            "category": op.category,
            "description": op.description,
        })

    return sorted(results, key=lambda x: x["name"])
