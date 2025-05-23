from setuptools import setup, find_packages

setup(
    name="self_healing_agents",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
