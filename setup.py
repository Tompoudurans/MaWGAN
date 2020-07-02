from setuptools import setup, find_packages

gans_requirements = [tensorflow,numpy]

tools_requirments = 

setup(
    name="gans",
    version="0.0.1",
    install_requires=requirements,
    packages=find_packages("src"),
    package_dir={"": "src"},
)

setup(
    name="tools",
    version="0.0.1",
    install_requires=requirements,
    packages=find_packages("src"),
    package_dir={"": "src"},
)
