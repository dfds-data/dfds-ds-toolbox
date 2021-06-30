# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
## Install locally
Make a virtual environment:
```shell
python -m venv venv
venv\Scripts\activate.bat
```
Install dependencies
```shell
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
pip install -e .
```

Run tests to see everything working
```shell
pytest
```

## Install this library in another repo
See [this guide in the wiki](https://dfds.visualstudio.com/Smart%20Data/_wiki/wikis/Smart-Data.wiki/2779/Installing-a-package-from-the-smartdata-artifact-feed)


# Contribute
We want this library to be useful across many data science projects. 
If you have some standard utilities that you keep using in your projects, please add them here and make a PR. 