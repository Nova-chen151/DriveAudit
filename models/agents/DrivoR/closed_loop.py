"""DrivoR inference on the six generated camera views for each round."""
from pathlib import Path


class Policy:
    def __init__(self,cfg):
        from driveaudit.agents.registry import load_adapter
        self.adapter=load_adapter('drivor').DrivoRAdapter(Path(cfg['policy_checkpoint']))
        self.run_output=cfg['run_output']
    def call(self,q):
        from driveaudit.closed_loop.image_contract import validate
        proof=validate(q,self.run_output)
        result=self.adapter.predict(q['model_input'])
        result['image_source_audit']=proof
        return result
