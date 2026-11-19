"""
Optional SwanLab logging for BC-HOI training/evaluation.

Usage:
    main.py ... --use_swanlab --swanlab_project bc-hoi --swanlab_mode online

Modes (SwanLab 0.10):
    online   - upload to the SwanLab cloud (requires `swanlab login` or SWANLAB_API_KEY)
    local    - run a local SwanLab dashboard
    offline  - write logs locally without the cloud
    disabled - accept API calls but do not record anything

If SwanLab is not installed, not logged in, or init fails for any reason, the
logger silently disables itself so that training/evaluation continues.
"""

import atexit

# Imported lazily in init_swanlab(): the huge BLIP-2/OPT weight download must
# not be affected by SwanLab's import-time code or internal safe.block errors.
swanlab = None

_enabled = False
_run = None


def is_enabled():
    return _enabled


def _to_python(value):
    """Convert torch scalars / numpy scalars to plain Python values."""
    if hasattr(value, 'detach'):
        try:
            value = value.detach().cpu().item()
        except Exception:
            return None
    elif hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return None


def _clean(data):
    clean = {}
    for key, value in data.items():
        value = _to_python(value)
        if value is not None:
            clean[key] = value
    return clean


def init_swanlab(args):
    global _enabled, _run, swanlab
    if not getattr(args, 'use_swanlab', False):
        return

    if swanlab is None:
        try:
            import swanlab as _swanlab
            swanlab = _swanlab
        except Exception as exc:  # pragma: no cover - optional dependency
            print('[SwanLab] package is not installed (%s); logging disabled.'
                  % exc)
            return

    mode = getattr(args, 'swanlab_mode', 'online')
    if mode == 'cloud':
        mode = 'online'

    init_kwargs = {
        'project': getattr(args, 'swanlab_project', 'bc-hoi'),
        'mode': mode,
        'config': vars(args),
    }
    name = getattr(args, 'swanlab_experiment', '')
    if name:
        init_kwargs['name'] = name
    log_dir = getattr(args, 'swanlab_logdir', '')
    if log_dir:
        init_kwargs['log_dir'] = log_dir

    try:
        _run = swanlab.init(**init_kwargs)
        _enabled = True
        atexit.register(finish)
        print('[SwanLab] logging enabled: project=%s mode=%s'
              % (init_kwargs['project'], mode))
    except Exception as exc:
        _enabled = False
        _run = None
        print('[SwanLab] init failed (%s); continuing without logging.' % exc)


def log(data, step=None):
    if not _enabled:
        return
    clean = _clean(data)
    if not clean:
        return
    try:
        if _run is not None:
            _run.log(clean, step=step)
        else:
            swanlab.log(clean, step=step)
    except Exception as exc:
        print('[SwanLab] log failed: %s' % exc)


def log_iter(data, epoch, iteration, loader_len, interval=10):
    """Log per-iteration metrics (throttled to every `interval` iterations)."""
    if not _enabled:
        return
    if interval > 1 and iteration % interval != 0:
        return
    step = epoch * max(loader_len, 0) + iteration
    log(data, step=step)


def finish():
    global _enabled, _run
    if not _enabled:
        return
    try:
        if _run is not None:
            _run.finish()
        else:
            swanlab.finish()
    except Exception as exc:
        print('[SwanLab] finish failed: %s' % exc)
    finally:
        _enabled = False
        _run = None
