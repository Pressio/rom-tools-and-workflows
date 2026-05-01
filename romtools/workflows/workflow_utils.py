from typing import Iterable
import os
import shutil
import subprocess

from romtools.hpc.dispatcher import Dispatcher


def create_empty_dir(dir_name: str, dispatcher: Dispatcher = None):
    '''
    Create empty directory

    If dispatcher is provided, the directory is created on the remote host.
    '''
    if dispatcher is not None:
        dispatcher.create_remote_directory(dir_name)
        return

    if os.path.exists(dir_name):
        pass
    else:
        os.makedirs(dir_name)


def setup_directory(source_dir: str,
                    target_dir: str,
                    files2link: Iterable = (),
                    files2copy: Iterable = ()):
    '''
    Create new directory and populate with files
    '''
    create_empty_dir(target_dir)
    for file in files2copy:
        shutil.copy(f'{source_dir}/{file}',
                    f'{target_dir}/{os.path.basename(file)}')
    for file in files2link:
        os.symlink(f'{source_dir}/{file}',
                   f'{target_dir}/{os.path.basename(file)}')


def run_model(module: str = None,
              pre_script: str = None,
              executable: str = 'bash',
              num_procs: int = 1,
              directory: str = '.',
              **kwargs):
    '''
    Execute external model
    '''
    execution_str = f'module load {module};' if module is not None else ''
    execution_str += pre_script if pre_script is not None else ''
    execution_str += f'mpirun -n {num_procs} {executable}'
    for flag, value in kwargs:
        execution_str += f' {flag} {value}'
    subprocess.run(execution_str, shell=True, check=True, cwd=directory)
