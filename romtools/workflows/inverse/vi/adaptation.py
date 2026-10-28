"""Accepted-iteration lifecycle for delayed MF-VI ROM adaptation."""

import numpy as np

from romtools.hpc.dispatchers import resolve_local_dispatcher


def stage_delayed_rom(state, builder, directory, training_limit, dispatcher):
    """Train only after acceptance, leaving the estimator's active ROM intact."""
    if state.get('rom_retraining_strategy') != 'delayed':
        return state
    state = dict(state)
    state['pending_rom_model'] = None
    if state.get('bootstrap_rom', False) or not state['rom_adaptation_required']:
        return state
    offline_directory = f'{directory}/rom_delayed_update'
    resolve_local_dispatcher(dispatcher).create_empty_dir(offline_directory)
    dirs = state['training_dirs'][-training_limit:]
    parameters = state['training_parameters'][-training_limit:].copy()
    qois = state['training_qois'][-training_limit:].copy()
    print(
        "Retraining ROM: "
        f"iteration_directory={directory}, rom_error={state.get('rom_error', np.nan):.5e}, "
        f"rom_tolerance={state.get('rom_tolerance', np.nan):.5e}, "
        f"training_samples={len(dirs)}, activation=next_iteration"
    )
    model = builder.build_from_training_dirs(offline_directory, dirs, parameters, qois)
    if model is None:
        raise ValueError('Delayed adaptation requires a replacement ROM model.')
    if state['rom_model'] is not None and model is state['rom_model']:
        raise ValueError('Delayed adaptation requires a distinct replacement ROM model.')
    state.update(
        pending_rom_model=model,
        pending_rom_training_dirs=list(dirs),
        pending_rom_training_parameters=parameters,
        pending_rom_training_qois=qois,
        rom_rebuilt_this_iteration=True,
    )
    return state


def activate_delayed_rom(state):
    """Select next iteration's model without changing the accepted estimates."""
    if state.get('pending_rom_model') is None:
        return state
    active = dict(state)
    active['rom_model'] = state['pending_rom_model']
    for suffix in ('dirs', 'parameters', 'qois'):
        active[f'rom_training_{suffix}'] = state[f'pending_rom_training_{suffix}']
    active['pending_rom_model'] = None
    return active


def mf_vi_restart_data(state):
    snapshot = {key: value for key, value in state.items()
                if key not in ('rom_model', 'pending_rom_model')}
    snapshot['has_pending_rom'] = state.get('pending_rom_model') is not None
    return {
        'rom_retraining_strategy': state['rom_retraining_strategy'],
        'gradient_gain_estimation_strategy': state['gradient_gain_estimation_strategy'],
        'hessian_gain_estimation_strategy': state['hessian_gain_estimation_strategy'],
        'mf_vi_state': np.array(snapshot, dtype=object),
    }


def validate_mf_vi_restart(restart, retraining_strategy, gain_strategy, training_limit=None, gain_num_folds=5, hessian_gain_strategy="same_sample", gain_sample_size=None, use_mfmc_control_variate=None, mfmc_fixed_gain=1.0):
    required = ('rom_retraining_strategy', 'gradient_gain_estimation_strategy', 'hessian_gain_estimation_strategy', 'mf_vi_state')
    if any(key not in restart for key in required):
        raise ValueError('restart_file is missing MF-VI strategy/state metadata.')
    if (str(restart['rom_retraining_strategy'].item()) != retraining_strategy
            or str(restart['gradient_gain_estimation_strategy'].item()) != gain_strategy
            or str(restart['hessian_gain_estimation_strategy'].item()) != hessian_gain_strategy):
        raise ValueError('restart_file MF-VI strategies do not match the current run.')
    snapshot = restart['mf_vi_state'].item()
    keys = {
        'has_pending_rom', 'optimizer_samples', 'parameter_samples_fom', 'gradient_mean',
        'elbo', 'rom_retraining_strategy', 'gradient_gain_estimation_strategy', 'hessian_gain_estimation_strategy', 'bootstrap_rom',
        'rom_adaptation_required', 'rom_training_dirs', 'rom_training_parameters',
        'rom_training_qois', 'pending_rom_training_dirs', 'pending_rom_training_parameters',
        'pending_rom_training_qois', 'gradient_gain_samples', 'hessian_gain_samples',
    }
    if (use_mfmc_control_variate is not None and not bool(use_mfmc_control_variate) and isinstance(snapshot, dict)
            and snapshot.get('mfmc_fixed_gain', 1.0) != mfmc_fixed_gain):
        raise ValueError('restart_file mfmc_fixed_gain does not match the current run.')
    if gain_strategy == 'independent':
        if not isinstance(snapshot, dict) or 'gain_sample_size' not in snapshot or 'gain_independent_data' not in snapshot or 'use_mfmc_control_variate' not in snapshot:
            raise ValueError('restart_file is missing independent gain metadata.')
        if snapshot['gain_sample_size'] != gain_sample_size:
            raise ValueError('restart_file gain_sample_size does not match the current run.')
        if use_mfmc_control_variate is not None and snapshot['use_mfmc_control_variate'] != use_mfmc_control_variate:
            raise ValueError('restart_file independent gain control-variate setting does not match the current run.')
        active = (not snapshot.get('bootstrap_rom', False)
                  and len(snapshot['parameter_samples_rom_only']) > 0
                  and snapshot['use_mfmc_control_variate'])
        data = snapshot['gain_independent_data']
        if data is not None:
            fields = ('optimizer_samples', 'parameter_samples', 'high_log_joint', 'low_log_joint',
                      'mean', 'std', 'correlation_cholesky', 'cholesky')
            if not isinstance(data, dict) or any(key not in data for key in fields):
                raise ValueError('restart_file has incomplete independent gain data.')
            n = len(data['optimizer_samples'])
            expected = len(snapshot['parameter_samples_fom']) if gain_sample_size is None else gain_sample_size
            if n != expected or any(len(data[key]) != n for key in fields[1:4]):
                raise ValueError('restart_file has inconsistent independent gain sample counts.')
            points = np.asarray(data['optimizer_samples'])
            if points.ndim != 2:
                raise ValueError('restart_file has invalid independent gain sample dimensions.')
            dimension = points.shape[1]
            shapes = dict(mean=(dimension,), std=(dimension,), cholesky=(dimension,dimension),
                          parameter_samples=(n,dimension), high_log_joint=(n,), low_log_joint=(n,))
            if (any(np.shape(data[key]) != shape for key,shape in shapes.items())
                    or not all(np.all(np.isfinite(data[key])) for key in shapes)
                    or not np.all(np.isfinite(points)) or np.any(np.asarray(data['std']) <= 0)):
                raise ValueError('restart_file has invalid independent gain data dimensions or values.')

        elif active:
            raise ValueError('restart_file is missing independent gain samples.')
    if gain_strategy == 'k_fold':
        from romtools.workflows.inverse.vi.gains import gain_folds
        if not isinstance(snapshot, dict) or snapshot.get('gain_num_folds') != gain_num_folds:
            raise ValueError('restart_file k-fold count metadata is missing or does not match the current run.')
        if 'parameter_samples_fom' not in snapshot:
            raise ValueError('restart_file has incomplete MF-VI state.')
        stored = snapshot.get('gain_fold_indices')
        expected = gain_folds(len(snapshot['parameter_samples_fom']), gain_num_folds)
        if stored is None or len(stored) != len(expected) or any(
                not np.array_equal(a, b) for a, b in zip(stored, expected)):
            raise ValueError('restart_file has incomplete k-fold membership metadata.')
    if gain_strategy == 'history':
        keys.update(('gain_distribution_mean', 'gain_distribution_std',
                     'gain_distribution_correlation_cholesky', 'gain_iteration_directory',
                     'gain_history_data', 'gain_training_batches', 'gain_history_training_limit',
                     'training_parameters'))
        if ('variational_distribution' in restart
                and str(restart['variational_distribution'].item()) == 'full_covariance'):
            keys.add('gain_distribution_cholesky')
    if not isinstance(snapshot, dict) or not keys.issubset(snapshot):
        raise ValueError('restart_file has incomplete MF-VI state.')
    if (snapshot['gradient_gain_estimation_strategy'] != gain_strategy
            or snapshot['hessian_gain_estimation_strategy'] != hessian_gain_strategy):
        raise ValueError('restart_file MF-VI state strategies do not match the current run.')
    if gain_strategy == 'history':
        if training_limit is not None and snapshot['gain_history_training_limit'] != training_limit:
            raise ValueError('restart_file history training-window limit does not match the current run.')
        batches = snapshot['gain_training_batches']
        row_keys = ('parameter_samples', 'optimizer_samples', 'high_log_joint',
                    'log_priors', 'log_transform_jacobian', 'sample_ids')
        batch_keys = set(row_keys) | {'mean', 'std', 'correlation_cholesky', 'source_directory'}
        if (not isinstance(batches, list) or not batches
                or any(not isinstance(batch, dict) or not batch_keys.issubset(batch) for batch in batches)):
            raise ValueError('restart_file has incomplete HF history-window metadata.')
        for batch in batches:
            count = len(batch['parameter_samples'])
            if count < 1 or any(len(batch[key]) != count for key in row_keys):
                raise ValueError('restart_file has inconsistent HF history-window sample arrays.')
            dimension = batch['optimizer_samples'].shape[1]
            mean, std = np.asarray(batch['mean']), np.asarray(batch['std'])
            if (mean.shape != (dimension,) or std.shape != (dimension,)
                    or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0)):
                raise ValueError('restart_file has invalid HF history sampling distributions.')
        ids = [value for batch in batches for value in batch['sample_ids']]
        if len(ids) != len(set(ids)) or len(ids) > snapshot['gain_history_training_limit']:
            raise ValueError('restart_file has invalid HF history-window membership.')
        window = np.concatenate([batch['parameter_samples'] for batch in batches])
        if not np.array_equal(window, snapshot['training_parameters'][-snapshot['gain_history_training_limit']:]):
            raise ValueError('restart_file HF history window does not match the retained training samples.')


def restore_mf_vi_state(restart, builder, directory, dispatcher):
    """Reconstruct both models; restoring a completed batch draws no samples."""
    state = dict(restart['mf_vi_state'].item())
    state['rom_model'] = None
    state['pending_rom_model'] = None
    rng_state = np.random.get_state()
    try:
        for prefix in ('rom', 'pending_rom'):
            if prefix == 'pending_rom' and not state.pop('has_pending_rom'):
                continue
            parameters = state[f'{prefix}_training_parameters']
            if parameters is None:
                continue
            offline_directory = f'{directory}/restore_{prefix}'
            resolve_local_dispatcher(dispatcher).create_empty_dir(offline_directory)
            state[f'{prefix}_model'] = builder.build_from_training_dirs(
                offline_directory, state[f'{prefix}_training_dirs'],
                parameters, state[f'{prefix}_training_qois'],
            )
    finally:
        np.random.set_state(rng_state)
    return state
