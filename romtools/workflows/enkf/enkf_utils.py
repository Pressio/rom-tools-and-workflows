from typing import Protocol, Iterable

import numpy as np

from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.models import QoiModel


def create_parameter_dict(
    parameter_names: Iterable[str],
    parameter_values: Iterable[float],
) -> dict:
    return dict(zip(parameter_names, parameter_values))


class Transformer(Protocol):

    def transform(self, my_array: np.ndarray) -> np.ndarray:
        '''Applies transform'''
        ...

    def inverse_transform(self, my_array: np.ndarray) -> np.ndarray:
        '''Applies inverse transform'''
        ...


class _Transformer():
    '''
    Supplies simple scalar transform and its inverse for feature scaling purposes.

    Does NOT perform operations in place, will return a new array
    This class conforms to the `Transformer` protocol.
    '''

    def __init__(self, a: float, b: float, c: float) -> None:
        self.__a = a
        self.__b = b
        self.__c = c

    def transform(self, my_array: np.ndarray) -> np.ndarray:
        '''Applies transform according to equation: my_array = a + (my_array + b) * c'''
        return self.__a + (my_array + self.__b) * self.__c

    def inverse_transform(self, my_array: np.ndarray) -> np.ndarray:
        '''Applies transform according to equation: my_array = (my_array - a) / c - b'''
        return (my_array - self.__a) / self.__c - self.__b


def create_minmax_transformer(
    minval: float,
    maxval: float,
    lbound: float = 0.0,
    ubound: float = 1.0,
) -> Transformer:
    '''
    Transforms to range [lbound, ubound].

    Requires assumed minimum (minval) and maximum (maxval) values of data to be transformed.
    Defaults to range [0.0, 1.0].
    '''
    a = lbound
    b = -minval
    c = (ubound - lbound) / (maxval - minval)
    transformer = _Transformer(a, b, c)
    return transformer


def multi_transform(
    in_array: np.ndarray,
    transformers: Iterable[Transformer],
    inverse: bool = False,
) -> np.ndarray:
    '''
    Operates multiple transformation across the last dimension of an array
    '''

    assert in_array.shape[-1] == len(transformers), "Unequal last dimension of my_array and number of transformers"

    out_array = in_array.copy()
    for trans_idx, transformer in enumerate(transformers):
        if inverse:
            out_array[..., trans_idx] = transformer.inverse_transform(in_array[..., trans_idx])
        else:
            out_array[..., trans_idx] = transformer.transform(in_array[..., trans_idx])

    return out_array


def multi_invers_transform(in_array: np.ndarray, transformers: Iterable[Transformer]) -> np.ndarray:
    '''
    Operates multiple inverse transformation across the last dimension of an array
    '''

    assert in_array.shape[-1] == len(transformers), "Unequal last dimension of my_array and number of transformers"

    out_array = in_array.copy()
    for trans_idx, transformer in enumerate(transformers):
        out_array[..., trans_idx] = transformer.transform(in_array[..., trans_idx])

    return out_array


def process_model_qois(
    param_inputs: np.ndarray,
    param_names: Iterable[str],
    model: QoiModel,
    outdir: str,
) -> np.ndarray:
    '''Run model given inputs, return QOIs'''

    create_empty_dir(outdir)
    parameter_dict = create_parameter_dict(param_names, param_inputs)
    model.populate_run_directory(outdir, parameter_dict)
    model.run_model(outdir, parameter_dict)

    return model.compute_qoi(outdir, parameter_dict)
