"""Six supported external model services."""
from pathlib import Path
from driveaudit_eval.config import dependencies_root

AGENTS = [('uniad', 'UniAD', 'uniad', []), ('vad', 'VAD', 'vad', []),
          ('drivor', 'DrivoR', 'drivoR', []),
          ('sparsedrivev2_v2', 'SparseDriveV2', 'sparsedrivev2', ['--version', 'v2']),
          ('diffusiondrivev2', 'DiffusionDriveV2', 'diffusiondrivev2', []),
          ('goalflow', 'GoalFlow', 'goalflow', [])]

FORMAL_AGENTS = tuple(item[0] for item in AGENTS)

def service_path(name):
    module = 'sparsedrivev2' if name.startswith('sparsedrivev2_') else name
    return Path(__file__).resolve().parent / 'adapters' / (module + '.py')


from dataclasses import dataclass

@dataclass(frozen=True)
class AgentRequirements:
    required_cameras: tuple
    rgb_required: bool = True
    calibration_required: bool = True
    trajectory_required: bool = True
    map_required: bool = False
    real_lidar_allowed: bool = False

    @property
    def lidar_required(self):
        """Compatibility for the explicitly selected multimodal research path."""
        return self.real_lidar_allowed

    @property
    def required_camera_count(self):
        return len(self.required_cameras)

    def missing(self, capabilities):
        missing = [name for name, required in [('rgb', self.rgb_required), ('calibration', self.calibration_required),
                   ('trajectory', self.trajectory_required), ('map', self.map_required), ('lidar', self.lidar_required)]
                   if required and not capabilities.get(name, False)]
        names = capabilities.get('camera_names')
        if (names is not None and not set(self.required_cameras).issubset(names)) or capabilities.get('camera_count', 0) < self.required_camera_count:
            missing.append('required_cameras')
        return missing

    def as_dict(self):
        from dataclasses import asdict
        return dict(asdict(self), required_camera_count=self.required_camera_count)


_FRONT = ('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')
_SIX = _FRONT + ('CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT')
REQUIREMENTS = {
    'uniad': AgentRequirements(_SIX),
    'vad': AgentRequirements(_SIX),
    'drivor': AgentRequirements(_FRONT+('CAM_BACK',)),
    'sparsedrivev2_v2': AgentRequirements(_FRONT),
    'diffusiondrivev2': AgentRequirements(_FRONT, calibration_required=False),
    'goalflow': AgentRequirements(_FRONT, calibration_required=False),
}


def requirements_for(name, sensor_mode='camera_only'):
    from dataclasses import replace
    if sensor_mode not in ('camera_only','original_multimodal'):
        raise ValueError('Unknown agent sensor protocol')
    return replace(REQUIREMENTS[name],real_lidar_allowed=sensor_mode=='original_multimodal' and name in ('goalflow','diffusiondrivev2'))


def write_support_matrix(scene_status, output):
    """A READY cell requires successful inference and evaluation, not just schema presence."""
    from .core import write_csv
    datasets=list(dict.fromkeys(s['dataset'] for s in scene_status))
    rows=[]
    for agent in dict.fromkeys(s['agent'] for s in scene_status):
        row={'agent':agent}
        for dataset in datasets:
            states=[s for s in scene_status if s['agent']==agent and s['dataset']==dataset]
            complete=[s for s in states if s['inference_status']=='completed']
            status='READY' if complete and len(complete)==len(states) and all(s['evaluation_status']=='completed' for s in states) else 'PARTIAL' if complete else 'BLOCKED'
            reasons=sorted({s['failure_reason'] for s in states if s.get('failure_reason')})
            row[dataset]=status
            row[dataset+'_reason']='; '.join(reasons) or ('' if status=='READY' else 'inference_not_completed')
        rows.append(row)
    write_csv(Path(output)/'CROSS_DATASET_AGENT_MATRIX.csv',rows)
    return rows
