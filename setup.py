from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="dfds_ds_toolbox",
    packages=find_packages(exclude=["*tests*"]),
    keywords=["dfds_ds_toolbox", "data science"],
    version="0.9.0",
    description="A collection of tools for data science used at DFDS.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Data Science Chapter at DFDS",
    author_email="urcha@dfds.com",
    install_requires=["scikit-learn", "pandas", "numpy", "matplotlib", "statsmodels", "rich"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
)
