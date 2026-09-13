"""Bound-handling adapters for EKI and MF-EKI.

The historical EKI implementations operate directly on their input parameter
ensemble. This module adds the same ``clip``/``transform`` choice used by VI
without duplicating the EKI algorithms. In transform mode, the legacy drivers
see an unconstrained optimizer-coordinate ensemble while FOM/ROM interfaces,
restart files, and returned samples remain in physical coordinates.
"""

from contextlib import contextmanager
from importlib import import_module
import inspect
import os
import tempfile

import numpy as np

from romtools.hpc.dispatchers import LocalDispatcher
from romtools.workflows.parameter_spaces import ParameterSpace


_eki_module = import_module("romtools.workflows.inverse.eki_drivers")
_mf_eki_module = import_module("romtools.workflows.inverse.mf_eki_drivers")
_vi_module = import_module("romtools.workflows.inverse.vi_drivers")

_legacy_run_eki = _eki_module.run_eki
_legacy_run_mf_eki = _mf_eki_module.run_mf_eki
_legacy_mf_eki_with_auto_rom = _mf_eki_module.mf_eki_with_auto_rom
_RUN_EKI_SIGNATURE = inspect.signature(_legacy_run_eki)
_RUN_MF_EKI_SIGNATURE = inspect.signature(_legacy_run_mf_eki)
_AUTO_MF_EKI_SIGNATURE = inspect.signature(_legacy_mf_eki_with_auto_rom)

# Reuse VI's implementation so ``transform`` has one definition across inverse
# workflows.
_normalize_bounded_parameter_handling = _vi_module._normalize_bounded_parameter_handling
_normalize_transform_map = _vi_module._normalize_transform_map
_transform_optimizer_to_parameter = _vi_module._transform_optimizer_to_parameter
_transform_parameter_to_optimizer = _vi_module._transform_parameter_to_optimizer

_PARAMETER_ARRAY_KEYS = (
    "parameter_samples",
    "parameter_samples_one",
    "parameter_samples_two",
    "training_parameters",
    "rom_training_parameters",
)


def _as_parameter_matrix(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[None, :], True
    if values.ndim != 2:
        raise ValueError("Parameter values must be a one- or two-dimensional array.")
    return values, False


def _map_optimizer_to_physical(values, mins, maxes, margin, transform_map):
    values, was_vector = _as_parameter_matrix(values)
    mapped = _transform_optimizer_to_parameter(
        values, mins, maxes, margin, transform_map
    )
    return mapped[0] if was_vector else mapped


def _map_physical_to_optimizer(values, mins, maxes, margin, transform_map):
    values, was_vector = _as_parameter_matrix(values)
    mapped = _transform_parameter_to_optimizer(
        values, mins, maxes, margin, transform_map
    )
    return mapped[0] if was_vector else mapped


def _validate_transform(parameter_space, mins, maxes, margin, transform_map):
    if mins is None or maxes is None:
        raise ValueError(
            "parameter_mins and parameter_maxes are required for "
            "bounded_parameter_handling='transform'."
        )
    mins = np.asarray(mins, dtype=float)
    maxes = np.asarray(maxes, dtype=float)
    dimensionality = parameter_space.get_dimensionality()
    if mins.ndim != 1 or mins.size != dimensionality:
        raise ValueError(
            "parameter_mins must be one-dimensional and match the parameter-space dimensionality."
        )
    if maxes.ndim != 1 or maxes.size != dimensionality:
        raise ValueError(
            "parameter_maxes must be one-dimensional and match the parameter-space dimensionality."
        )
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxes)):
        raise ValueError("Transform-based bound handling requires finite parameter bounds.")
    if not np.all(maxes > mins):
        raise ValueError(
            "All parameter_maxes entries must be greater than parameter_mins entries "
            "for bounded_parameter_handling='transform'."
        )
    if not 0.0 <= margin < 0.5:
        raise ValueError("transform_interior_margin must be in [0.0, 0.5).")
    return mins, maxes, _normalize_transform_map(transform_map)


class _OptimizerParameterSpace(ParameterSpace):
    """Preserve the requested physical initial draw, then inverse-transform it."""

    def __init__(self, parameter_space, mins, maxes, margin, transform_map):
        self._parameter_space = parameter_space
        self._mins = mins
        self._maxes = maxes
        self._margin = margin
        self._transform_map = transform_map

    def get_names(self):
        return self._parameter_space.get_names()

    def get_dimensionality(self):
        return self._parameter_space.get_dimensionality()

    def generate_samples(self, number_of_samples: int, seed=None):
        physical = np.asarray(
            self._parameter_space.generate_samples(number_of_samples, seed),
            dtype=float,
        ).copy()
        # Match historical initialization if a parameter space draws outside
        # the requested EKI bounds, but never clip transformed updates.
        physical = np.minimum(physical, self._maxes[None, :])
        physical = np.maximum(physical, self._mins[None, :])
        return _map_physical_to_optimizer(
            physical, self._mins, self._maxes, self._margin, self._transform_map
        )


class _PhysicalParameterQoiModel:
    """Present physical parameter dictionaries to a model evaluated by z-space EKI."""

    def __init__(self, model, names, mins, maxes, margin, transform_map):
        self._model = model
        self._names = list(names)
        self._mins = mins
        self._maxes = maxes
        self._margin = margin
        self._transform_map = transform_map

    def _physical_dict(self, parameter_sample):
        optimizer_values = np.asarray(
            [parameter_sample[name] for name in self._names], dtype=float
        )
        physical_values = _map_optimizer_to_physical(
            optimizer_values,
            self._mins,
            self._maxes,
            self._margin,
            self._transform_map,
        )
        mapped = dict(parameter_sample)
        mapped.update(zip(self._names, physical_values))
        return mapped

    def populate_run_directory(self, run_directory, parameter_sample):
        return self._model.populate_run_directory(
            run_directory, self._physical_dict(parameter_sample)
        )

    def run_model(self, run_directory, parameter_sample):
        return self._model.run_model(
            run_directory, self._physical_dict(parameter_sample)
        )

    def compute_qoi(self, run_directory, parameter_sample):
        return self._model.compute_qoi(
            run_directory, self._physical_dict(parameter_sample)
        )


class _PhysicalParameterQoiModelBuilder:
    """Present physical MF-EKI training parameters to a user ROM builder."""

    def __init__(self, builder, names, mins, maxes, margin, transform_map):
        self._builder = builder
        self._names = list(names)
        self._mins = mins
        self._maxes = maxes
        self._margin = margin
        self._transform_map = transform_map

    def build_from_training_dirs(
            self, offline_data_dir, training_data_dirs, training_parameters, training_qois):
        physical_training_parameters = _map_optimizer_to_physical(
            training_parameters,
            self._mins,
            self._maxes,
            self._margin,
            self._transform_map,
        )
        model = self._builder.build_from_training_dirs(
            offline_data_dir,
            training_data_dirs,
            physical_training_parameters,
            training_qois,
        )
        return _PhysicalParameterQoiModel(
            model,
            self._names,
            self._mins,
            self._maxes,
            self._margin,
            self._transform_map,
        )


def _npz_scalar_string(value):
    if value is None:
        return None
    value = np.asarray(value)
    if value.size != 1:
        return None
    return str(value.reshape(-1)[0]).strip().lower()


def _convert_parameter_arrays(arrays, mins, maxes, margin, transform_map, mapper):
    converted = dict(arrays)
    for key in _PARAMETER_ARRAY_KEYS:
        if key in converted:
            converted[key] = mapper(
                converted[key], mins, maxes, margin, transform_map
            )
    return converted


def _restart_ensemble(arrays):
    if "parameter_samples" in arrays:
        return np.asarray(arrays["parameter_samples"], dtype=float)
    if "parameter_samples_one" in arrays and "parameter_samples_two" in arrays:
        return np.vstack((arrays["parameter_samples_one"], arrays["parameter_samples_two"]))
    return None


def _ensemble_covariance(samples):
    if samples is None or samples.ndim != 2 or samples.shape[0] < 2:
        return None
    anomalies = samples - np.mean(samples, axis=0)[None, :]
    return anomalies.T @ anomalies / (samples.shape[0] - 1)


class _PhysicalRestartDispatcher:
    """Write physical-coordinate transformed-EKI restarts through a dispatcher."""

    def __init__(self, dispatcher, mins, maxes, margin, transform_map):
        self._dispatcher = dispatcher
        self._mins = mins
        self._maxes = maxes
        self._margin = margin
        self._transform_map = transform_map

    def __getattr__(self, name):
        return getattr(self._dispatcher, name)

    def np_savez(self, path, **arrays):
        if path.endswith("restart.npz"):
            arrays = _convert_parameter_arrays(
                arrays,
                self._mins,
                self._maxes,
                self._margin,
                self._transform_map,
                _map_optimizer_to_physical,
            )
            arrays.update(
                bounded_parameter_handling=np.asarray("transform"),
                parameter_sample_coordinates=np.asarray("physical"),
                transform_map=np.asarray(self._transform_map),
                transform_interior_margin=np.asarray(self._margin, dtype=float),
                rejuvenation_reference_covariance_coordinates=np.asarray("optimizer"),
            )
        self._dispatcher.np_savez(path, **arrays)


@contextmanager
def _optimizer_restart_file(restart_file, mins, maxes, margin, transform_map):
    """Materialize a legacy/transformed physical restart in optimizer coordinates."""
    if restart_file is None:
        yield None
        return

    with np.load(restart_file, allow_pickle=True) as restart_data:
        arrays = {key: restart_data[key] for key in restart_data.files}

    saved_handling = _npz_scalar_string(arrays.get("bounded_parameter_handling"))
    saved_coordinates = _npz_scalar_string(arrays.get("parameter_sample_coordinates"))
    if saved_handling == "transform":
        saved_map = _npz_scalar_string(arrays.get("transform_map"))
        saved_margin_array = arrays.get("transform_interior_margin")
        saved_margin = (
            float(np.asarray(saved_margin_array).reshape(-1)[0])
            if saved_margin_array is not None else 0.0
        )
        if saved_map is not None and _normalize_transform_map(saved_map) != transform_map:
            raise ValueError("Restart transform_map does not match the requested transform_map.")
        if not np.isclose(saved_margin, margin):
            raise ValueError(
                "Restart transform_interior_margin does not match the requested value."
            )
        if saved_coordinates not in (None, "physical", "optimizer"):
            raise ValueError(
                f"Unsupported restart parameter_sample_coordinates '{saved_coordinates}'."
            )
        if saved_coordinates != "optimizer":
            arrays = _convert_parameter_arrays(
                arrays, mins, maxes, margin, transform_map, _map_physical_to_optimizer
            )
    else:
        # Legacy clipped restarts store physical samples and a physical fallback
        # covariance. Convert samples and rebuild that fallback covariance.
        arrays = _convert_parameter_arrays(
            arrays, mins, maxes, margin, transform_map, _map_physical_to_optimizer
        )
        covariance = _ensemble_covariance(_restart_ensemble(arrays))
        if covariance is not None:
            arrays["rejuvenation_reference_covariance"] = covariance

    arrays.update(
        bounded_parameter_handling=np.asarray("transform"),
        parameter_sample_coordinates=np.asarray("optimizer"),
        transform_map=np.asarray(transform_map),
        transform_interior_margin=np.asarray(margin, dtype=float),
    )

    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez(path, **arrays)
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _bound_call(signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _transform_signature(function):
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    additions = [
        inspect.Parameter(
            "bounded_parameter_handling",
            inspect.Parameter.KEYWORD_ONLY,
            default="clip",
            annotation=str,
        ),
        inspect.Parameter(
            "transform_interior_margin",
            inspect.Parameter.KEYWORD_ONLY,
            default=0.0,
            annotation=float,
        ),
        inspect.Parameter(
            "transform_map",
            inspect.Parameter.KEYWORD_ONLY,
            default="sigmoid",
            annotation=str,
        ),
    ]
    insert_at = next(
        (i for i, p in enumerate(parameters) if p.kind == inspect.Parameter.KEYWORD_ONLY),
        len(parameters),
    )
    parameters[insert_at:insert_at] = additions
    return signature.replace(parameters=parameters)


def _transform_context(arguments, handling, margin, transform_map):
    if handling == "clip":
        return None
    mins, maxes, transform_map = _validate_transform(
        arguments["parameter_space"],
        arguments["parameter_mins"],
        arguments["parameter_maxes"],
        margin,
        transform_map,
    )
    names = list(arguments["parameter_space"].get_names())
    return mins, maxes, transform_map, names


def run_eki(*args,
            bounded_parameter_handling="clip",
            transform_interior_margin=0.0,
            transform_map="sigmoid",
            **kwargs):
    """Run EKI with historical clipping or an unconstrained bound transform."""
    handling = _normalize_bounded_parameter_handling(bounded_parameter_handling)
    arguments = _bound_call(_RUN_EKI_SIGNATURE, args, kwargs)
    context = _transform_context(
        arguments, handling, transform_interior_margin, transform_map
    )
    if context is None:
        return _legacy_run_eki(**arguments)

    mins, maxes, transform_map, names = context
    model = arguments["model"]
    parameter_space = arguments["parameter_space"]
    dispatcher = arguments["dispatcher"] or LocalDispatcher()
    restart_file = arguments["restart_file"]

    arguments.update(
        model=_PhysicalParameterQoiModel(
            model, names, mins, maxes, transform_interior_margin, transform_map
        ),
        parameter_space=_OptimizerParameterSpace(
            parameter_space, mins, maxes, transform_interior_margin, transform_map
        ),
        parameter_mins=None,
        parameter_maxes=None,
        dispatcher=_PhysicalRestartDispatcher(
            dispatcher, mins, maxes, transform_interior_margin, transform_map
        ),
    )
    with _optimizer_restart_file(
            restart_file, mins, maxes, transform_interior_margin, transform_map) as restart:
        arguments["restart_file"] = restart
        optimizer_samples, qois = _legacy_run_eki(**arguments)
    return (
        _map_optimizer_to_physical(
            optimizer_samples, mins, maxes, transform_interior_margin, transform_map
        ),
        qois,
    )


def run_mf_eki(*args,
               bounded_parameter_handling="clip",
               transform_interior_margin=0.0,
               transform_map="sigmoid",
               **kwargs):
    """Run MF-EKI with historical clipping or an unconstrained bound transform."""
    handling = _normalize_bounded_parameter_handling(bounded_parameter_handling)
    arguments = _bound_call(_RUN_MF_EKI_SIGNATURE, args, kwargs)
    context = _transform_context(
        arguments, handling, transform_interior_margin, transform_map
    )
    if context is None:
        return _legacy_run_mf_eki(**arguments)

    mins, maxes, transform_map, names = context
    model = arguments["model"]
    builder = arguments["rom_model_builder"]
    parameter_space = arguments["parameter_space"]
    dispatcher = arguments["dispatcher"] or LocalDispatcher()
    restart_file = arguments["restart_file"]

    arguments.update(
        model=_PhysicalParameterQoiModel(
            model, names, mins, maxes, transform_interior_margin, transform_map
        ),
        rom_model_builder=_PhysicalParameterQoiModelBuilder(
            builder, names, mins, maxes, transform_interior_margin, transform_map
        ),
        parameter_space=_OptimizerParameterSpace(
            parameter_space, mins, maxes, transform_interior_margin, transform_map
        ),
        parameter_mins=None,
        parameter_maxes=None,
        dispatcher=_PhysicalRestartDispatcher(
            dispatcher, mins, maxes, transform_interior_margin, transform_map
        ),
    )
    with _optimizer_restart_file(
            restart_file, mins, maxes, transform_interior_margin, transform_map) as restart:
        arguments["restart_file"] = restart
        optimizer_samples, qois = _legacy_run_mf_eki(**arguments)
    return (
        _map_optimizer_to_physical(
            optimizer_samples, mins, maxes, transform_interior_margin, transform_map
        ),
        qois,
    )


def _auto_rom_builder(parameter_space, rom_type, rom_args):
    rom_args = {} if rom_args is None else dict(rom_args)
    rom_type = rom_type.strip().lower()
    if rom_type == "gp":
        return _mf_eki_module.GaussianProcessQoiModelBuilderWithTrainingData(
            parameter_names=parameter_space.get_names(),
            pod_energy_fraction=rom_args.get("pod_energy_fraction", 0.999999),
            max_pod_modes=rom_args.get("max_pod_modes"),
            kernel=rom_args.get("kernel"),
            noise_variance=rom_args.get("noise_variance"),
            auto_noise_variance=rom_args.get("auto_noise_variance", False),
            noise_variance_fraction=rom_args.get("noise_variance_fraction", 1e-6),
            tune_hyperparameters=rom_args.get("tune_hyperparameters", False),
            length_scale_grid=rom_args.get("length_scale_grid"),
            signal_variance_grid=rom_args.get("signal_variance_grid"),
            normalize_parameters=rom_args.get("normalize_parameters", False),
            normalize_targets=rom_args.get("normalize_targets", False),
        )
    if rom_type in ("nn", "neural_network", "neural-network"):
        return _mf_eki_module.NeuralNetworkQoiModelBuilderWithTrainingData(
            parameter_names=parameter_space.get_names(),
            pod_energy_fraction=rom_args.get("pod_energy_fraction", 0.999999),
            max_pod_modes=rom_args.get("max_pod_modes"),
            network_config=rom_args.get("network_config"),
            lipschitz_config=rom_args.get("lipschitz_config"),
            normalize_parameters=rom_args.get("normalize_parameters", True),
            normalize_targets=rom_args.get("normalize_targets", True),
        )
    raise ValueError(
        f"Unsupported rom_type '{rom_type}'. Supported options are 'gp' and 'nn'."
    )


def mf_eki_with_auto_rom(*args,
                         bounded_parameter_handling="clip",
                         transform_interior_margin=0.0,
                         transform_map="sigmoid",
                         **kwargs):
    """Run automatic-ROM MF-EKI with clipping or transformed bounds."""
    handling = _normalize_bounded_parameter_handling(bounded_parameter_handling)
    arguments = _bound_call(_AUTO_MF_EKI_SIGNATURE, args, kwargs)
    if handling == "clip":
        return _legacy_mf_eki_with_auto_rom(**arguments)

    rom_model_builder = _auto_rom_builder(
        arguments["parameter_space"], arguments["rom_type"], arguments["rom_args"]
    )
    arguments.pop("rom_type")
    arguments.pop("rom_args")
    arguments["rom_model_builder"] = rom_model_builder
    return run_mf_eki(
        bounded_parameter_handling=handling,
        transform_interior_margin=transform_interior_margin,
        transform_map=transform_map,
        **arguments,
    )


run_eki.__signature__ = _transform_signature(_legacy_run_eki)
run_mf_eki.__signature__ = _transform_signature(_legacy_run_mf_eki)
mf_eki_with_auto_rom.__signature__ = _transform_signature(_legacy_mf_eki_with_auto_rom)

__all__ = ["run_eki", "run_mf_eki", "mf_eki_with_auto_rom"]
