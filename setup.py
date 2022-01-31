from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="dfds_ds_toolbox",
    packages=find_packages(exclude=["*tests*"]),
    keywords=["dfds_ds_toolbox", "data science"],
    version="0.8.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ida Friisberg, Uriel Chareca, Dennis Hansen, Pablo Canada",
    author_email="idfri@dfds.com, urcha@dfds.com, dhans@dfds.com, pacaa@dfds.com",
    install_requires=["scikit-learn", "pandas", "numpy", "matplotlib"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
