from setuptools import find_packages, setup

setup(
    name="brouhaha",
    # namespace_packages=["pyannote"],
    # version=version,
    packages=find_packages(),
    description="Pyannote extension for SNR and C50 predictions",
    long_description_content_type="text/markdown",
    author="Marianne Métais",
    url="https://github.com/marianne-m/brouhaha-vad",
)