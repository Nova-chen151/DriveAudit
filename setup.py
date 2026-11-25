"""Keep native runtime trees intact when installing from a wheel."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithNativeSources(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        assets = Path(self.build_lib) / "driveaudit" / "_assets"
        for name in ("configs", "models"):
            shutil.copytree(root / name, assets / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".git"))


setup(cmdclass={"build_py": BuildWithNativeSources})
