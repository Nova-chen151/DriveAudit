"""DriveAudit public API; importing an adapter never loads evaluator dependencies."""


def evaluate(config=None, **kwargs):
    """Evaluate a config; see driveaudit_eval.evaluation.evaluate for selection options."""
    from driveaudit_eval.evaluation import evaluate as run
    return run(config, **kwargs)


__all__ = ['evaluate']
