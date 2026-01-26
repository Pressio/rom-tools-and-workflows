from pathlib import Path

from setuptools import find_packages, setup


def read_version() -> str:
    return Path("version.txt").read_text(encoding="utf-8").strip()


setup(
    name="romtools",
    version=read_version(),
    python_requires=">=3.8",
    packages=find_packages(where="."),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "romtools-format=romtools.workflows.formatting:main",
        ]
    },
)
