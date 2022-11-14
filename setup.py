from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent

with open(here / "requirements.txt") as file:
    requirements = file.read().splitlines()

with open(here / "README.md") as file:
    readme = file.read()

setup(
    name="brouhaha",
    version='0.9.0',
    packages=find_packages(),
    description="Pyannote extension for SNR and C50 predictions",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Marianne Métais",
    url="https://github.com/marianne-m/brouhaha-vad",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
)
