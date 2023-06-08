# CIFAR & MNIST Joint Classification

### Requirements

Our implementation was tested in Python&nbsp;3.10.10. This program uses the [PyTorch](https://pytorch.org/) neural network framework, version&nbsp;1.13.0.  For the package requirements, see `requirements.txt.

We recommend running our program in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).  Once your virtual environment is created and *active*, run the following in this directory:

```
pip install --user --upgrade pip
pip install -r requirements.txt
```

### Viewing Influence Rankings

We use the [Weights & Biases](https://wandb.ai/) package `wandb` to visualize ranking results.  W&B is free to [register](https://wandb.ai/login?signup=true) and use.  

You should enable your W&B account on your local system before running the program.  This is done by calling:

```
pip install wandb
wandb login
```

If you want to run without W&B, set the variable `USE_WANDB` in `poison/_config.py` to `False`.


### Running the Program

To run the program, simply call `./run.sh` in this directory.  All datasets are downloaded automatically and installed in the directory `.data`.  
