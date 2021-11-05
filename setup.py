from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="ds_toolbox",
    packages=find_packages(exclude=["*tests*"]),
    keywords=["ds_toolbox", "data science"],
    version="0.5.0",
    long_description=long_description,
    author="Ida Friisberg, Uriel Chareca, Dennis Hansen, Pablo Canada",
    author_email="idfri@dfds.com, urcha@dfds.com, dhans@dfds.com, pacaa@dfds.com",
    install_requires=["scikit-learn", "pandas", "numpy", "statsmodels", "rich", "matplotlib"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Data Scientists",
        "Programming Language :: Python 3.7",
        "Programming Language :: Python 3.8",
        "Programming Language :: Python 3.9",
    ],
)
