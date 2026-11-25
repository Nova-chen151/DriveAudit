"""Expose the bundled model integrations under driveaudit.agents.<Model>."""
from driveaudit.resources import resource_root

__path__.append(str(resource_root() / "models" / "agents"))
