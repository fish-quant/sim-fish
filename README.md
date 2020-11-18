# Sim-FISH

[![License](https://img.shields.io/badge/license-BSD%203--Clause-green)](https://github.com/fish-quant/big-fish/blob/master/LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

**Sim-FISH** is a python package to **simulate smFISH images**. The package allows the user simulate and localize spots, cells and nuclei. The ultimate goal is to provide toy images to **experiment, train and evaluate smFISH statistical analysis**.


## Installation

### Dependencies

Sim-FISH requires Python 3.6 or newer. Additionally, it has the following dependencies:

- numpy (== 1.16.0)
- scipy (== 1.4.1)
- scikit-learn (== 0.20.2)
- scikit-image (== 0.14.2)
- matplotlib (== 3.0.2)
- pandas (== 0.24.0)
- mrc (== 0.1.5)

Updated dependencies might break.

### Virtual environment

To avoid dependency conflicts, we recommend the the use of a dedicated [virtual](https://docs.python.org/3.6/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment.  In a terminal run the command:

```bash
conda create -n simfish_env python=3.6
source activate simfish_env
```

We recommend two options to then install Sim-FISH in your virtual environment.

#### Download the package from PyPi (not available yet)



#### Clone package from Github

Clone the project's [Github repository](https://github.com/fish-quant/sim-fish) and install it manually with the following commands:

```bash
git clone git@github.com:fish-quant/sim-fish.git
cd sim-fish
pip install .
```

## Usage



## Support

If you have any question relative to the repository, please open an [issue](https://github.com/fish-quant/sim-fish/issues). You can also contact [Arthur Imbert](mailto:arthur.imbert@mines-paristech.fr).

## Roadmap (suggestion)

Version 0.1.0:
- I/O routines.
- Random spot simulation in 2D and 3D.
- Clustered spot simulation in 2D and 3D.
- Allow benchmark and valuation pipeline.

Version 0.Y.0:
- Noise simulation.
- Localized spot simulation in 2D. 
- Cell and nucleus simulation in 2D.
- Localized spot simulation in 3D. 
- Cell and nucleus simulation in 3D.

Version 1.0.0:
- Complete code coverage.
- Add sphinx documentation.

## Development

### Source code

You can access the latest sources with the commands:

```bash
git clone git@github.com:fish-quant/sim-fish.git
cd sim-fish
git checkout develop
```

### Contributing

[Pull requests](https://github.com/fish-quant/sim-fish/pulls) are welcome. For major changes, please open an [issue](https://github.com/fish-quant/sim-fish/issues) first to discuss what you would like to change.
