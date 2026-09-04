"""
Model ordering and index conventions for the hybrid MF UQ workflow.

This module is the single source of truth for how estimators are ordered and
how pairs of estimators are flattened into the lists that `MFMC` consumes.
Every consumer imports from here rather than re-deriving the ordering; before
this module existed the same convention was spelled out in five places, each
carrying a comment asking the reader to keep it in sync with the others.

Notation map (writeup -> code)
------------------------------
The writeup and the implementation use different names for the same objects.

    writeup                         code
    ------------------------------- --------------------------------------
    m           number of LF models  n_lofi = n_aux + n_active
    k           number of trainable  n_active (== len(rom_model_builders))
    M = {0..m}  all estimators       full index space, size n_models
    F           fixed estimators     FOM (index 0) + aux models
    T           trainable estimators the ROMs
    omega, q    a trainable index    t, q (0-based, within the ROM block)
    Q_0         high-fidelity QoI    fom_qois
    Q_i, i in F auxiliary QoI        aux_qois_list[i]
    s_omega     ROM basis size       s / s_active entry for ROM t
    B_omega     pilot basis grid     pilot_basis_grids[t]
    N_p         pilot sample count   num_pilot / pilot_sample_size
    B           resampling replicates combination index k (max_combinations)
    h           holdout size         min_pair_validation_size (see caveat in
                                     Pilot.set_ROM_correlation_labels)
    p_hat_{i,q} pilot correlation    fom_aux_corrs, aux_aux_corrs,
                                     fom_rom_corrs_list, aux_rom_corrs_list,
                                     rom_rom_corrs
    w_tilde_i   normalized cost      normalized_aux_times,
                                     normalized_rom_times_list
    P(s)        correlation surrogate corr_matrix_fn / AHMatrixCorrelationSurrogate
    C_target    normalized budget    budget / budget_list entries
    r           oversampling ratios  r (optimizer variable)

Equation map (writeup -> implementation)
----------------------------------------
    (3), (4), (14)  R^2_ACV             MFMC.set_objective_and_constraint
    (5)             fixed model cost    PilotSampler._compute_stats
    (6)             fixed-fixed corr    Pilot.estimate_pairwise_correlations
    (7), (8)        fixed-trainable     Pilot.estimate_pairwise_correlations
    (9)             trainable cost      PilotSampler._normalized_rom_times
    (10), (11)      trainable-trainable Pilot.estimate_ROM_ROM_correlation
    (12)            AH fixed point      WarmStartedArchakovHansenMap
    (13)            split of P(s)       MFMC._split_corr_matrix
    (15)            hybrid ACV problem  MFMC.solve
    (16)            discrete solution   MFMC.discretize
    (18)            validation problem  run_hybrid_mfuq Step 4
    Sec. 4.3.1      cost surrogates     fit_cost_polynomial
    Sec. 4.3.2      sigmoid surrogates  fit_sigmoid
    Sec. 4.3.3      AH matrix surrogate archakov_hansen.py

Two index spaces
----------------
Full model space (size `n_models`), the ordering of the correlation matrix P:

    0                      FOM (Q_0)
    1 .. n_aux             auxiliary models
    n_aux+1 .. n_aux+k     trainable ROMs, in rom_model_builders order

Low-fidelity space (size `n_lofi`), the ordering of `cost_list`,
`hf_corr_list`, and the rows/columns of the LF-LF matrix C:

    0 .. n_aux-1           auxiliary models
    n_aux .. n_aux+k-1     trainable ROMs, in rom_model_builders order

The LF space is the full space with the FOM row/column removed, so
`lf_slot + 1 == model_slot`.

Pair flattening
---------------
Pairs are always enumerated in strict lower-triangular row-major order: row i
ascending, then column j < i ascending. This is `np.tril_indices(n, -1)`
order. It is the order `MFMC.build_C` reads `lf_corr_list` back in, and the
order the pilot writes `aux_aux_corrs` in. Note that it differs from
`itertools.combinations` order once n >= 4.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sizes and slots
# ---------------------------------------------------------------------------

def n_models(n_aux, n_active):
    """Size of the full model index space (FOM + aux + trainable ROMs)."""
    return 1 + n_aux + n_active


def n_lofi(n_aux, n_active):
    """Size of the low-fidelity index space (aux + trainable ROMs)."""
    return n_aux + n_active


def aux_model_slot(i):
    """Full-space index of auxiliary model i."""
    return i + 1


def rom_model_slot(n_aux, t):
    """Full-space index of trainable ROM t."""
    return 1 + n_aux + t


def rom_lf_slot(n_aux, t):
    """Low-fidelity-space index of trainable ROM t."""
    return n_aux + t


def rom_state_offset(n_active, t):
    """
    Negative offset of ROM t's coordinate within an expanded state vector.

    Cost and scalar correlation surrogates are called with the expanded state
    s (length n_lofi), whose trailing n_active entries are the ROM basis
    sizes in rom_model_builders order.
    """
    return -(n_active - t)


# ---------------------------------------------------------------------------
# Pair enumeration
# ---------------------------------------------------------------------------

def tril_indices(n):
    """Strict lower-triangular index pairs of an n x n matrix, row-major."""
    return np.tril_indices(n, -1)


def tril_position(i, j):
    """
    Position of entry (i, j), i > j, in the flat row-major lower-triangular
    ordering. Inverse of `tril_indices` for a single entry.
    """
    return i * (i - 1) // 2 + j


def aux_pairs(n_aux):
    """
    Yield (pair_index, i, j) for every auxiliary-auxiliary pair i > j, in the
    order `aux_aux_corrs` is stored in. Empty when n_aux < 2.
    """
    index = 0
    for i in range(n_aux):
        for j in range(i):
            yield index, i, j
            index += 1


def matrix_pairs(n_aux, n_active):
    """
    Yield (i, j, kind, t, q) for every off-diagonal entry i > j of the full
    n_models x n_models correlation matrix.

    kind is one of:
        "fixed_fixed"  both estimators fixed (FOM-aux or aux-aux); t, q None.
        "fixed_rom"    fixed estimator j, trainable ROM t = i; q None.
        "rom_rom"      trainable ROMs t = j and q = i, with t < q.
    """
    n_fixed = 1 + n_aux

    for i in range(n_models(n_aux, n_active)):
        for j in range(i):
            if i < n_fixed:
                yield i, j, "fixed_fixed", None, None
            elif j < n_fixed:
                yield i, j, "fixed_rom", i - n_fixed, None
            else:
                yield i, j, "rom_rom", j - n_fixed, i - n_fixed


def lf_pairs(n_aux, n_active):
    """
    Yield (i, j, kind, t, q) for every off-diagonal entry i > j of the LF-LF
    correlation matrix, in the flat order `lf_corr_list` is stored in and
    `MFMC.build_C` reads back.

    kind is one of:
        "aux_aux"  auxiliary pair; t, q None.
        "aux_rom"  auxiliary model j and trainable ROM t = i - n_aux; q None.
        "rom_rom"  trainable ROMs t = j - n_aux and q = i - n_aux, t < q.
    """
    for i in range(n_lofi(n_aux, n_active)):
        for j in range(i):
            if i < n_aux:
                yield i, j, "aux_aux", None, None
            elif j < n_aux:
                yield i, j, "aux_rom", i - n_aux, None
            else:
                yield i, j, "rom_rom", j - n_aux, i - n_aux
