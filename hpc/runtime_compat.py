"""Fail early if HPC uses a different Isaac generation from the reference code."""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import sys


def get_isaacsim_version():
    """Read pip metadata or the VERSION shipped with a standalone installation."""
    try:
        return importlib.metadata.version('isaacsim'), 'pip metadata: isaacsim'
    except importlib.metadata.PackageNotFoundError:
        pass

    candidates = []
    spec = importlib.util.find_spec('isaacsim')
    if spec is not None:
        package_dirs = list(spec.submodule_search_locations or ())
        if spec.origin:
            package_dirs.append(str(Path(spec.origin).parent))
        for directory in package_dirs:
            package = Path(directory)
            # pip's package root; binary install's python_packages/isaacsim.
            candidates.extend((package / 'VERSION', package.parent.parent / 'VERSION'))
    for name in ('ISAAC_PATH', 'ISAAC_SIM_PATH', 'ISAAC_SIM_ROOT'):
        if os.environ.get(name):
            candidates.append(Path(os.environ[name]) / 'VERSION')
    # Kit's embedded Python lives below the installation root.
    candidates.extend(parent / 'VERSION' for parent in Path(sys.executable).resolve().parents)
    candidates.append(Path('/isaac-sim/VERSION'))

    for path in dict.fromkeys(candidates):
        if path.is_file():
            lines = path.read_text(encoding='utf-8-sig').splitlines()
            version = lines[0].strip() if lines else ''
            if not re.match(r'^\d+\.\d+\.\d+', version):
                raise RuntimeError(f'Invalid Isaac Sim VERSION file: {path}: {version!r}')
            return version, str(path)
    raise RuntimeError('Cannot determine Isaac Sim version: pip metadata is absent and no VERSION file '
                       'was found. Check /isaac-sim/VERSION or set ISAAC_PATH to the actual '
                       'Isaac Sim installation root. Checked: ' + ', '.join(map(str, candidates)))


def validate_versions(runtime):
    required = {'python': '3.12', 'isaacsim': '6.0',
                'isaaclab_framework': '3.0', 'rsl_rl': '5.4'}
    for name, prefix in required.items():
        value = runtime.get(name, '')
        if '.'.join(value.split('.')[:2]) != prefix:
            raise RuntimeError(f'{name} {value!r} does not match this project ({prefix}.x); '
                               'use the same IsaacLab checkout and container as cross_stick')


def check_runtime():
    import isaaclab
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import setup_preset_cli
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    del add_launcher_args, launch_simulation, setup_preset_cli, handle_deprecated_rsl_rl_cfg
    version_path = Path(isaaclab.__file__).resolve().parents[3] / 'VERSION'
    if not version_path.is_file():
        raise RuntimeError(f'IsaacLab VERSION not found: {version_path}; mount the full IsaacLab checkout')
    isaacsim_version, isaacsim_version_source = get_isaacsim_version()
    runtime = {'python': platform.python_version(),
               'isaacsim': isaacsim_version,
               'isaacsim_version_source': isaacsim_version_source,
               'isaaclab_framework': version_path.read_text().strip(),
               'isaaclab_source': str(Path(isaaclab.__file__).resolve()),
               'rsl_rl': importlib.metadata.version('rsl-rl-lib'),
               'torch': importlib.metadata.version('torch')}
    validate_versions(runtime)
    print('[RUNTIME]', runtime, flush=True)
    return runtime
