from pathlib import Path

from setuptools import find_packages, setup

# What packages are required for this module to be executed?
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"

with open("VERSION") as f:
    _version_ = f.read().strip()
with open("PACKAGENAME") as f:
    _name_ = f.read().strip()


def list_reqs(fname="basic.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()


setup(
    name=_name_,
    version=_version_,
    packages=find_packages(exclude=["docs", "tests"]),
    package_data={"": ["*.txt", "*.yaml"]},
    install_requires=list_reqs(),
    extras_require={
        "dev": list_reqs(fname="dev.txt"),
        "serve": list_reqs(fname="serve.txt"),
        "onnx": list_reqs(fname="onnx.txt"),
        "compression": list_reqs(fname="compression.txt"),
        "torch": list_reqs(fname="torch.txt"),
        "tensorflow": list_reqs(fname="tensorflow.txt"),
        "llm": list_reqs(fname="llm.txt"),
    },
)
