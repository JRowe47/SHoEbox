"""Public policy selection, without reinterpreting historical constructors."""
from .model import Model as HistoricalModel, _library

DEFAULT_POLICY = "e8"  # Verified numeric/lifecycle/speed gates and nominal API cap.


def _select(policy, options):
    if policy not in (None, "v19", "e8"):
        raise ValueError("policy must be 'v19' or 'e8'")
    if policy is not None:
        return policy
    # Any explicit architecture selection keeps its pre-integration meaning.
    if set(options) - {"parameters", "seed"}:
        return "v19"
    return DEFAULT_POLICY


class Model(HistoricalModel):
    """Versioned SHoEbox. Explicit architectural constructors remain historical."""
    def __new__(cls, *, policy=None, **options):
        if cls is Model and _select(policy, options) == "e8":
            from .e8.model import E8Model
            return object.__new__(E8Model)
        return object.__new__(cls)

    def __init__(self, *, policy=None, **options):
        if policy == "v19":
            if options.get("core") not in (None, "minimal"):
                raise ValueError("policy='v19' requires core='minimal'")
            options["core"] = "minimal"
        super().__init__(**options)

    @classmethod
    def _wrap(cls, handle, profile):
        obj = object.__new__(Model)
        obj._profile = profile
        obj._lib, obj._handle = _library(profile), handle
        obj._check_pointer()
        return obj

    @classmethod
    def loads(cls, data):
        if data[:8] == b"SHOEME01":
            from .e8.model import E8Model
            return E8Model.loads(data)
        return super().loads(data)

    @classmethod
    def import_e8_research(cls, data):
        """Explicit conversion; returns (model, source/destination hash receipt)."""
        from .e8.model import E8Model
        return E8Model.import_research(data)


def fresh_model(**kwargs):
    return Model(**kwargs)

# Preserve constructor introspection used by historical experiment tooling.
import inspect as _inspect
Model.__signature__ = _inspect.signature(HistoricalModel).replace(parameters=[
    *_inspect.signature(HistoricalModel).parameters.values(),
    _inspect.Parameter('policy', _inspect.Parameter.KEYWORD_ONLY, default=None),
])
