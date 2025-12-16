import os
from setuptools import setup, find_packages

if not os.path.isdir("build"):
    os.makedirs("build")

setup(
    name="func",
    version="0.1.0",
    description="Utilities for fragment-based molecule generation",
    author="A. Sato",
    packages=find_packages(where="src", include=["func", "func.*"]),
    package_dir={"": "src"},
    include_package_data=False,
    command_options={
    "egg_info": {
        "egg_base": ("setup.py", "build"),
        }
    }
)
