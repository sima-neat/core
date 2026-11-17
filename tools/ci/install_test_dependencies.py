"""Install declared runtime/dev dependencies without rebuilding the installed wheel."""
import subprocess
import sys
import tomllib
from pathlib import Path


def install(project_file):
    with project_file.open("rb") as source:
        project = tomllib.load(source)["project"]
    requirements = [*project["dependencies"],
                    *project["optional-dependencies"]["dev"]]
    if not requirements:
        raise ValueError("Test dependency declaration must not be empty")
    subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)


if __name__ == "__main__":
    install(Path(__file__).resolve().parents[2] / "pyproject.toml")
