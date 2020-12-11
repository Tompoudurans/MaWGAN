from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)
setup(
    name="ganrunner",
    version="0.0.4",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
