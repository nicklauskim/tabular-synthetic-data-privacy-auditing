![tests](https://github.com/alan-turing-institute/privacy-sdg-toolbox/actions/workflows/ci.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/privacy-sdg-toolbox/badge/?version=latest)](https://privacy-sdg-toolbox.readthedocs.io/en/latest/?badge=latest)

# Adversarial Privacy Auditing of Tabular Synthetic Data Generators (Using TAPAS Framework)

Evaluating the privacy of synthetic data with an adversarial toolbox. This code utilizes the TAPAS toolbox presented in [the associated paper](https://arxiv.org/abs/2211.06550). The details below on installation, etc. are taken directly from their repository (https://github.com/alan-turing-institute/privacy-sdg-toolbox).

[Documentation.](https://privacy-sdg-toolbox.readthedocs.io/en/latest/index.html#)


## Quickstart

**For a quick, over-arching view of the workflow for running a privacy audit on a synthetic data generator, take a look at the main file [here](https://github.com/nicklauskim/Tabular-Synthetic-Data-Privacy-Auditing/blob/main/src/groundhog_audit.py).**

## Reference

If you use this toolbox for a scientific publication, we kindly ask you to reference the paper:

	Houssiau, F., Jordon, J., Cohen, S.N., Daniel, O., Elliott, A., Geddes, J., Mole, C., Rangel-Smith, C. and Szpruch, L., 2022. _TAPAS: a toolbox for adversarial privacy auditing of synthetic data._

In `BibTex`:

```
@article{houssiau2022tapas,
  title={TAPAS: a toolbox for adversarial privacy auditing of synthetic data},
  author={Houssiau, F and Jordon, J and Cohen, SN and Daniel, O and Elliott, A and Geddes, J and Mole, C and Rangel-Smith, C and Szpruch, L},
  year={2022},
  publisher={Neural Information Processing Systems Foundation}
}
```


## Direct Installation

### Requirements
The framework and its building blocks have been developed and tested under Python 3.9.


#### Poetry installation
To mimic our environment exactly, we recommend using `poetry`. To install poetry (system-wide), follow the instructions [here](https://python-poetry.org/docs/).

Then run
```
poetry install
```
from inside the project directory. This will create a virtual environment (default `.venv`), that can be accessed by running `poetry shell`, or in the usual way (with `source .venv/bin/activate`).

#### Pip installation (includes command-line tool)

It is also possible to install from pip:
```
pip install git+https://github.com/alan-turing-institute/privacy-sdg-toolbox
```

Doing so installs a command-line tool, `tapas`, somewhere in your path. (Eg, on
a MacOS system with pip installed via homebrew, the tool ends up in a homebrew
bin director.) 
