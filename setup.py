from setuptools import setup, find_packages

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)

setup(
    name="ganrunner",
    version="0.0.3",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
