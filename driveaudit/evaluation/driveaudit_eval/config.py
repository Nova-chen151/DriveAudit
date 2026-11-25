"""Load public presets and explicit user overrides without model imports."""
from copy import deepcopy
from pathlib import Path
import os


def project_root():
    source = Path(__file__).resolve().parents[1]
    return source if (source / 'pyproject.toml').is_file() else Path.cwd()


def dependencies_root():
    """Local dependencies by default; relative overrides follow the project root."""
    directory = Path(os.path.expandvars(os.environ.get('DRIVEAUDIT_DEPENDENCIES', 'dependencies'))).expanduser()
    return (project_root() / directory).resolve()


def load_config(path, profile=None):
    import yaml
    def merge(target, values):
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = deepcopy(value)
    def read(source):
        if isinstance(source, dict):
            return deepcopy(source)
        source = Path(source).resolve()
        values = yaml.safe_load(source.read_text(encoding='utf-8'))
        if not isinstance(values, dict):
            raise ValueError('Config must be a YAML mapping')
        advanced = values.pop('advanced', None)
        inherited = read(source.parent / advanced) if advanced else {}
        merge(inherited, values)
        return inherited
    # Resolve inheritance before presets, so inherited defaults cannot mask a
    # research selection. User overrides are applied exactly once at the end.
    user = read(path)
    configured_preset = user.pop('defaults', None)
    preset = profile or configured_preset
    config = {}
    if preset is not None:
        if preset not in ('leaderboard', 'research'):
            raise ValueError('defaults must be leaderboard or research')
        package = Path(__file__).resolve().parent
        config = yaml.safe_load((package / 'defaults.yaml').read_text())
        if preset == 'research':
            merge(config, yaml.safe_load((package / 'research/defaults.yaml').read_text()))
    merge(config, user)
    if profile:
        config.setdefault('evaluation', {})['profile'] = profile
        config['evaluation'].setdefault('consistency', {})['profile'] = 'core' if profile == 'leaderboard' else 'research'
    from driveaudit_eval.metrics.registry import TRS_POLICY
    options = config.get('evaluation', {}).get('core_metrics', {}).get('trs')
    if options is not None:
        expected = dict(policy=TRS_POLICY['name'], protocol=TRS_POLICY['version'],
                        **{k: v for k, v in TRS_POLICY.items() if k not in ('name', 'version')})
        for key, value in expected.items():
            if key in options and options[key] != value:
                raise ValueError('TRS policy mismatch: ' + key)
        options.update(expected)
    policy = config.pop('policy', TRS_POLICY['name'])
    if policy != TRS_POLICY['name']:
        raise ValueError('Unknown formal TRS policy')
    data = config.pop('data', {})
    if options is not None:
        if 'root' in data: options['input_root'] = data['root']
        if 'reference_data' in data: options['reference_root'] = data['reference_data']
    for section in ('video', 'trajectory', 'consistency', 'agent'):
        settings = config.setdefault('evaluation', {}).setdefault(section, {})
        if 'scenes' in data:
            settings['scene_paths'] = [str(Path(data.get('root', '.')) / scene) for scene in data['scenes']]
    if 'generated_video' in data:
        config['evaluation']['video'].setdefault('paths', {})['generated_root'] = data['generated_video']
    if 'reference_data' in data:
        config['evaluation']['video'].setdefault('paths', {})['real_reference_root'] = data['reference_data']
    agents = config.get('agents')
    if isinstance(agents, list):
        config['agents'] = {'include': agents}
        config['evaluation']['agents'] = {'include': agents}
    output = config.get('output', {})
    if 'dir' in output:
        directory = Path(output.pop('dir'))
        output.update(root=str(directory.parent), run_name=directory.name)
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            value = value.replace('${PROJECT_ROOT}', str(project_root()))
            value = value.replace('${DEPENDENCIES_ROOT}', str(dependencies_root()))
            return os.path.expandvars(os.path.expanduser(value))
        return value
    return expand(config)
