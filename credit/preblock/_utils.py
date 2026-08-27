def _parse_variable_selection(variable_list: list, state_dict: dict, data_types: list = None) -> list:
    """
    The goal of this function is to expand partial variables in `variable_list` based on the keys in `state_dict`.
    For example, if `variable_list` contains "ERA5/prognostic/3d", then this function should expand that partial name
    to include every variable in `state_dict` that has the matching partial name.

    Args:
        variable_list (list): A list of partial or full variable names. An empty list means that all variables should be used.
        state_dict (dict): A dictionary of the data state following the CREDIT convention, nested as
        `state_dict[data_type][source][var_name]`. The data types (input, target, prediction) are the
        highest level key, then the source (e.g. "era5"), then the full variable names
        (e.g. `"era5/prognostic/3d/temperature"`).
        data_types (list): A list of data types to include. If None, all data types will be included.

    Returns:
        list: The full variable names from `state_dict` matched by `variable_list`, with duplicates
        removed and order preserved.
    """
    # Default to every data type present in the state.
    if data_types is None:
        data_types = list(state_dict.keys())

    # Collect every full variable name across the selected data types and sources,
    # preserving first-seen order and dropping duplicates (the same variable may
    # appear under more than one data type).
    all_variables = []
    for data_type in data_types:
        for source in state_dict.get(data_type, {}).values():
            for var_name in source:
                if var_name not in all_variables:
                    all_variables.append(var_name)

    # An empty selection means "use everything".
    if not variable_list:
        return all_variables

    # Expand each (possibly partial) name: a variable matches if it equals the
    # entry or sits beneath it in the "/"-delimited hierarchy.
    selected = []
    for partial in variable_list:
        for var_name in all_variables:
            if var_name == partial or var_name.startswith(partial + "/"):
                if var_name not in selected:
                    selected.append(var_name)
    return selected


def _flatten_spatial_tensors(state_dict: dict, spatial_variables: list) -> tuple[dict, dict]:
    """Return a copy of `state_dict` with tensors for `spatial_variables` flattened to (batch, -1).

    Grid-wise ("spatial") scalers store one statistic per gridpoint rather than
    per vertical level, so their channel dimension has to be the flattened
    spatial axis instead of the level axis. Folding every non-batch dimension
    (level, time, and the spatial dims) into a single trailing axis produces a
    rank-2 tensor, where the `channels_first` (dim=1) and `channels_last`
    (dim=-1) conventions point at the same axis — so the flattened tensor
    transforms correctly under either scaler setting with no extra branching.

    Requires the level and time dims (indices 1 and 2) to be singleton, since
    these blocks always operate on one forecast step at a time; flattening a
    >1 level or time dim would blend distinct slices into the same per-gridpoint
    statistic.

    Args:
        state_dict (dict): Nested batch dict (any depth) whose leaves are tensors.
        spatial_variables (list): Full variable keys (not partial paths — callers
            must resolve prefixes via `_parse_variable_selection` first) to flatten.
            An empty list is a no-op.

    Returns:
        tuple[dict, dict]: A copy of `state_dict` with the matched tensors
        flattened, and a `{var_key: original_shape}` map to pass to
        `_unflatten_spatial_tensors` afterward.
    """
    if not spatial_variables:
        return state_dict, {}
    spatial_set = set(spatial_variables)
    original_shapes = {}

    def _walk(d):
        new_d = {}
        for key, value in d.items():
            if isinstance(value, dict):
                new_d[key] = _walk(value)
            elif key in spatial_set:
                if value.shape[1] != 1 or value.shape[2] != 1:
                    raise ValueError(
                        f"Spatial scaling for '{key}' requires singleton level and time dims "
                        f"(dims 1 and 2), got shape {tuple(value.shape)}."
                    )
                original_shapes[key] = value.shape
                new_d[key] = value.reshape(value.shape[0], -1)
            else:
                new_d[key] = value
        return new_d

    return _walk(state_dict), original_shapes


def _unflatten_spatial_tensors(state_dict: dict, original_shapes: dict) -> dict:
    """Return a copy of `state_dict` with tensors restored to the shapes recorded by `_flatten_spatial_tensors`."""
    if not original_shapes:
        return state_dict

    def _walk(d):
        new_d = {}
        for key, value in d.items():
            if isinstance(value, dict):
                new_d[key] = _walk(value)
            elif key in original_shapes:
                new_d[key] = value.reshape(original_shapes[key])
            else:
                new_d[key] = value
        return new_d

    return _walk(state_dict)


def accelerate_bridgescaler_column_order() -> bool:
    """Make ``get_column_order`` O(n) instead of O(n^2). Idempotent; returns True if applied.

    bridgescaler resolves the scaler-to-input column mapping with two list scans::

        assert all(var in self.x_columns_ for var in x_in_columns)
        [self.x_columns_.index(item) for item in x_in_columns if item in self.x_columns_]

    Each ``in``/``index`` is O(n), so the pair is O(n^2) in the column count. That is
    harmless for a per-level scaler (32 columns) and ruinous for a per-gridpoint one:
    CAMulator normalizes surface pressure by a 2-D (lat, lon) field, giving 55296 columns
    and ~3e9 comparisons per call. Measured at 41 s per call, six calls per optimizer step,
    single-threaded on the CPU with the GPU at 0% -- roughly 500 s of an 8-minute batch.

    Caching a ``{name: index}`` dict on the scaler removes it from the profile. The indices
    are bit-identical: same values, same order, same duplicate handling (dict keeps the
    first occurrence, exactly as ``list.index`` returns it).
    """
    try:
        from bridgescaler import distributed_tensor as _dt
    except ImportError:
        return False

    base = getattr(_dt, "DBaseScalerTensor", None)
    if base is None or getattr(base, "_credit_fast_column_order", False):
        return False

    def get_column_order(self, x_in_columns):
        cache = self.__dict__.get("_credit_col_index")
        cols = self.x_columns_
        if cache is None or cache[0] is not cols:
            index = {}
            for i, name in enumerate(cols):
                index.setdefault(name, i)  # first occurrence, matching list.index
            cache = (cols, index)
            self.__dict__["_credit_col_index"] = cache
        index = cache[1]
        missing = [v for v in x_in_columns if v not in index]
        assert not missing, (
            f"Some input variables not in scaler x_columns. Missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
        return [index[item] for item in x_in_columns]

    base.get_column_order = get_column_order
    base._credit_fast_column_order = True
    return True


def restore_warning_filters_after_bridgescaler() -> bool:
    """Undo bridgescaler's global ``warnings.simplefilter("always")``. Returns True if applied.

    ``bridgescaler.distributed_tensor`` calls ``warnings.simplefilter("always")`` at module
    import, which wipes the host application's warning configuration process-wide and makes
    every warning print on every occurrence. CREDIT sets ``filterwarnings("ignore")`` when
    ``train_gen2`` is imported, but bridgescaler is imported later -- when the preblocks are
    built -- so it wins.

    The visible result is one "Input data lacks variable names" per variable per scaler call:
    about 400 per training batch, ~200k lines in a single chained PBS job, which buries the
    log messages that matter.

    That warning is not diagnostic here. ``scale_var_dict`` walks the nested state dict and
    looks each scaler up **by variable key**, asserting the key exists, so the pairing that
    could actually go wrong is checked by name. What bridgescaler cannot validate is the
    order *within* one variable's channel axis -- the 32 levels of T, the 55296 gridpoints of
    PS -- and both sides take that order from the same channel schema.

    This resets the filters and reinstates ``ignore``, matching what the application asked
    for. Warnings CREDIT raises through ``logger.warning`` are unaffected.
    """
    import warnings

    warnings.resetwarnings()
    warnings.filterwarnings("ignore")
    return True
