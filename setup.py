from setuptools import setup, find_packages

setup(
    name="Mawgan",
    version="0.0.5",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
