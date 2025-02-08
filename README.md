# Masterthesis-Self-Supervised Laplace Approximation

This Repository contains the code to reproduce the results from my Masterthesis.
It benchmarks the Self-Supervised Laplace Approximation Algorithm, first introduced by Rodemann et. al., against numerous UQ-Methods in different settings (conjugate prior, heteroscedastic noise, realworld data).

# Get started

Create a new venv

```
python -m venv .venv
```

Activate the venv

```
.\.venv\Scripts\Activate.ps1
```

(or use the shell version)

Install Poetry

```
pip install poetry
```

Install dependency (optionally with dev. append --with dev)

```
poetry install
```

Run the experiments

```
. .\scripts\run_experiments.ps1
```

(optionally use the bash script)

    Important

    Running the experiments will overwrite the existing experiment outputs.
    We seed everything, so the resulting graphics and tables should be the same

# Project Structure

- experiments
- src
- scripts

The **experiments** folder contains the experiments. It is structured as follows

- conjugate prior: Contains the conjugate prior experiments, namely: Bayesian Linear Model, Normal-Normal Model, Poisson-Gamma Model. Each experiment contains sub-experiments run under different data-generating processes or priors (or seeds, or no. of observations)
- heteroscedastic regression: Contains the results and the code for running the heteroscedastic regression experiment. It further contains the HMC class which was used for this experiment
- uciml: Contains the code and results on real-world data from UCIMLRepo.

The **src** folder contains next to utilities, the UCIMLRepo Datamodule and service classes for handling UQ-Methods like MFVI and VI (as well as basic MLPs) the (A)SSLA classes inside the ssl folder.

**scripts** contains the scripts to run the experiments using a single command (see above).
