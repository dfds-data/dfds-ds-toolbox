from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="ds_toolbox",
    packages=["ds-toolbox"],
    keywords=["ds_toolbox", "data science"],
    version="0.0.1",
    long_description=long_description,
    author="Ida Friisberg, Uriel Chareca , Dennis Hansen",
    author_email="idfri@dfds.com, urcha@dfds.com, dhans@dfds.com",
    install_requires=[
        "scikit-learn",
        "pandas"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Data Scientists",
        "Programming Language :: Python 3.7",
        "Programming Language :: Python 3.8",
        "Programming Language :: Python 3.9"
    ],
)