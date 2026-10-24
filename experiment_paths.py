"""Resolve experiment inputs without depending on the shell's working directory."""

from pathlib import Path
import os
import re
import shlex

PROJECT_ROOT = Path(__file__).resolve().parent


def format_command(arguments, *, powershell=None):
    """Produce a copyable preview, including paths with spaces or apostrophes."""
    if powershell is None:
        powershell = os.name == 'nt'
    if powershell:
        return '& ' + ' '.join("'" + str(arg).replace("'", "''") + "'" for arg in arguments)
    return shlex.join(str(arg) for arg in arguments)


def resolve_input_path(value, *, checkpoint=False):
    path = Path(value).expanduser()
    roots = [path] if path.is_absolute() else [path, PROJECT_ROOT / path]
    candidates = list(roots)
    if checkpoint:
        for candidate in roots:
            parts = list(candidate.parts)
            for index, part in enumerate(parts):
                match = re.fullmatch(r'seed_(\d+)', part)
                if match:
                    alternate = parts.copy()
                    alternate[index] = f'seed_{int(match.group(1)):02d}'
                    candidates.append(Path(*alternate))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return roots[-1].resolve()


def manifest_entry_path(value, manifest):
    """Accept new relative entries and relocate older absolute manifests."""
    path = Path(value)
    root = Path(manifest).resolve().parent
    if not path.is_absolute():
        return root / path
    for marker in ('configs', 'runs'):
        if marker in path.parts:
            index = path.parts.index(marker)
            relocated = root.joinpath(*path.parts[index:])
            if relocated.exists() or not path.exists():
                return relocated
    return path


def missing_checkpoints_message(missing):
    rows = '\n'.join(f'  {label}: {path}' for label, path in missing)
    return (
        f'Missing trained checkpoints:\n{rows}\n'
        f'Working directory: {Path.cwd()}\nProject directory: {PROJECT_ROOT}\n'
        'The standard training directory is seed_00 for training seed 0.\n'
        'GitHub source downloads do not contain .pth files; --write-plan only writes '
        'a plan. Copy compatible trained checkpoints here or train them first.\n'
        'To evaluate the three rule policies without trained models, run paper_experiments.py and omit all '
        '--model/--idqn-model/--qmix-model/--mappo-model options.\n'
        'No training or evaluation has started.'
    )
