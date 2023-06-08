# Target Identification and Renormalized Influence

[![docs](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ZaydH/certified-sparse/blob/main/LICENSE)

This repository contains the source code for reproducing the results of the [CCS'22](https://dl.acm.org/doi/abs/10.1145/3548606.3559335) paper "Identifying a Training-Set Attack's Target Using Renormalized Influence Estimation".

* Authors: [Zayd Hammoudeh](https://zaydh.github.io/) and [Daniel Lowd](https://ix.cs.uoregon.edu/~lowd/)
* Link to Paper: [Arxiv](https://arxiv.org/abs/2302.11628)

## Running the Program

Each task is divided into a different subprogram.  Inside the corresponding source directory, call:

`python driver.py ConfigFile`

where `ConfigFile` is one of the `yaml` configuration files in folder `configs`.

### Requirements

Our implementation was tested in Python&nbsp;3.10.10.  For the full requirements, see `requirements.txt`.

We recommend running our program in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).  Once your virtual environment is created and *active*, run the following in the `src` directory:

```
pip install --user --upgrade pip
pip install -r requirements.txt
```

## License

[MIT](https://github.com/ZaydH/certified-sparse/blob/main/LICENSE)

## Citation

```
@inproceedings{Hammoudeh:2022:TargetIdentification,
    author    = {Zayd Hammoudeh and
                 Daniel Lowd},
    title     = {Identifying a Training-Set Attack's Target Using Renormalized Influence Estimation},
    booktitle = {Proceedings of the 2022 {ACM} {SIGSAC} Conference on Computer and Communications Security},
    series = {{CCS}’22},
    year = {2022},
}
```
