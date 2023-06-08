# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "InfFuncTensors",
    "get_model_params",
    "compute_gradients",
    "compute_influences",
    "get_loss_with_weight_decay",
    "get_loss_without_wd",
]

import dataclasses
from typing import Callable, List, Union, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np

import torch
from torch import LongTensor, Tensor
import torch.nn as nn
import torch.utils.data

from . import general_utils as utils
from .. import _config as config


@dataclasses.dataclass
class InfFuncTensors:
    inf_base: Tensor = None
    inf_sim: Tensor = None

    s_test: List[Tensor] = None

    ds_ids: LongTensor = None
    bd_ids: LongTensor = None


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_loss_with_weight_decay(device: torch.device, n_gpu: int, model: torch.nn.Module,
                               f_loss: Callable,
                               x: Tensor, y: Tensor, weight_decay: Optional[float],
                               weight_decay_ignores: Optional[List[str]],
                               get_activations: bool = False) \
        -> Union[Tensor, Tuple[Tensor, Tensor]]:

    loss, outs = get_loss_without_wd(device=device, model=model, x=x, y=y, f_loss=f_loss)

    # # model outputs are always tuple in transformers (see doc)
    # loss = outputs[0]

    if n_gpu > 1:
        # mean() to average on multi-gpu parallel training
        loss = loss.mean()

    # In PyTorch, weight-decay loss and gradients are calculated in
    # optimizers rather in nn.Module, so we have to manually specify
    # this for the loss here.
    if weight_decay is not None:
        no_decay = (
            weight_decay_ignores
            if weight_decay_ignores
            is not None else [])

        # noinspection PyUnresolvedReferences
        weight_decay_loss = torch.cat([
            p.square().view(-1)
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]).sum() * weight_decay
        loss = loss + weight_decay_loss

    if get_activations:
        return loss, outs
    return loss


def get_loss_without_wd(device: torch.device, model: torch.nn.Module,
                        x: Tensor, y: Tensor, f_loss: Callable) -> Tuple[Tensor, Tensor]:
    r""" Calculates teh loss excluding weight decay """
    # model.train()
    x, y = x.to(device), y.to(device)
    acts = model.forward(x)
    # loss = F.cross_entropy(acts, y)
    loss = f_loss(acts, y)
    return loss, acts.detach()


def get_model_params(model: nn.Module, params_filter: Optional[List[str]] = None) -> List[Tensor]:
    r"""
    :param model: Model whose parameters will be returned
    :param params_filter: Name of any parameters to exclude
    :return: List of the parameters
    """
    if params_filter is None:
        params_filter = []
    return [param for name, param in model.named_parameters() if name not in params_filter]


def compute_gradients(device: torch.device, n_gpu: int, model: torch.nn.Module,
                      f_loss: Callable,
                      x: Tensor, y: Tensor,
                      params_filter: Optional[List[str]], weight_decay: Optional[float],
                      weight_decay_ignores: Optional[List[str]],
                      create_graph: bool = True,
                      return_loss: bool = False, return_acts: bool = False):
    r"""

    :param device: CUDA device used to calculate the data
    :param n_gpu:
    :param model:
    :param f_loss: Loss function
    :param x:
    :param y:
    :param params_filter:
    :param weight_decay:
    :param weight_decay_ignores:
    :param create_graph: If \p True, enables construction of derivative graph. This allows
                         computing higher order derivative products.
    :param return_loss: If \p True, return \p x's loss as well as the gradient
    :param return_acts: If \p True, return the output activations
    :return:
    """
    if params_filter is None:
        params_filter = []

    model.zero_grad()
    # Single loss value
    loss, acts = get_loss_with_weight_decay(model=model, device=device, n_gpu=n_gpu,
                                            f_loss=f_loss,
                                            x=x, y=y, weight_decay=weight_decay,
                                            weight_decay_ignores=weight_decay_ignores,
                                            get_activations=True)

    inputs = get_model_params(model=model, params_filter=params_filter)

    # create_graph=True: Enables construction of derivative graph. This allows computing higher
    # order derivative products.
    grad = torch.autograd.grad(outputs=loss, inputs=inputs, create_graph=create_graph)
    # Return a single value if no additional returns specified
    if not return_loss and not return_acts:
        return grad
    # Build the components of the output
    rets = []
    if return_loss:
        rets.append(loss)
    if return_acts:
        rets.append(acts)
    rets.append(grad)
    return tuple(rets)


def compute_hessian_vector_products(model: torch.nn.Module, device: torch.device, n_gpu: int,
                                    f_loss: Callable,
                                    x: Tensor, y: Tensor,
                                    vectors: torch.FloatTensor, params_filter: Optional[List[str]],
                                    weight_decay: Optional[float],
                                    weight_decay_ignores: Optional[List[str]])\
        -> Tuple[torch.Tensor, ...]:
    r"""

    :param model:
    :param device: \p torch Device where the calculations will be performed
    :param n_gpu: Number of GPUs in the system
    :param f_loss: Loss function
    :param x:
    :param y:
    :param vectors:
    :param params_filter: Name of any neural network parameters not to be considered in the
                          Hessian vector product gradient.
    :param weight_decay: Weight decay hyperparameter to ensure accurate calculation of the loss
                         since \p torch's weight decay is handled by the optimizer.  Essentially
                         L2 regularization.
    :param weight_decay_ignores: Any parameters (e.g., bias) not considered in the weight decay
                                 (L2) regularization calculation.
    :return:
    """
    if params_filter is None:
        params_filter = []
    # Output is the parameterized gradient \nabla_{\theta} L(x, y)
    grad_tuple = compute_gradients(model=model, device=device, n_gpu=n_gpu, f_loss=f_loss,
                                   x=x, y=y, params_filter=params_filter,
                                   weight_decay=weight_decay,
                                   weight_decay_ignores=weight_decay_ignores)

    model.zero_grad()
    grad_grad_inputs = [param for name, param in model.named_parameters()
                        if name not in params_filter]

    # inputs: Inputs w.r.t. which the gradient is taken.  Simply the parameters since
    #         Hessian is w.r.t. \theta, i.e., $\nabla^{2}_{\theta}$
    # outputs: Outputs of the function.  This function is being differentiated.  Here we are
    #          differentiating the \nabla_{\theta} L(x,y).  This yields the Hessian.
    # create_graph=False (unlike in method compute_gradients) since no need to create graph as only
    #                    taking the Hessian and not higher order terms.
    grad_grad_tuple = torch.autograd.grad(outputs=grad_tuple, inputs=grad_grad_inputs,
                                          grad_outputs=vectors, only_inputs=True)

    return grad_grad_tuple


def compute_s_test(model: torch.nn.Module, n_gpu: int, device: torch.device, f_loss: Callable,
                   test_dl: torch.utils.data.DataLoader,
                   train_data_loader: torch.utils.data.DataLoader,
                   params_filter: Optional[List[str]],
                   weight_decay: Optional[float], weight_decay_ignores: Optional[List[str]],
                   damp: float, scale: float, num_samples: Optional[int] = None,
                   verbose: bool = True) -> List[torch.Tensor]:

    all_x, all_y = [], []
    for xs, ys in test_dl:
        all_x.append(xs)
        all_y.append(ys)
    te_x, te_y = torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)
    v = compute_gradients(model=model, n_gpu=n_gpu, device=device, f_loss=f_loss,
                          x=te_x, y=te_y,
                          params_filter=params_filter,
                          weight_decay=weight_decay, weight_decay_ignores=weight_decay_ignores)

    # Technically, it's hv^-1
    last_estimate = list(v).copy()
    cumulative_num_samples = 0
    # with tqdm(total=num_samples) as pbar:
    #     for data_loader in train_data_loaders:
    #         for i, inputs in enumerate(data_loader):
    with tqdm(total=num_samples, disable=config.QUIET) as pbar:
        for i, inputs in enumerate(train_data_loader):
            x, y = inputs[0], inputs[1]
            this_estim = compute_hessian_vector_products(model=model, n_gpu=n_gpu, device=device,
                                                         vectors=last_estimate,  # noqa
                                                         f_loss=f_loss,
                                                         x=x, y=y, params_filter=params_filter,
                                                         weight_decay=weight_decay,
                                                         weight_decay_ignores=weight_decay_ignores)
            # Recursively calculate h_estimate
            # https://github.com/dedeswim/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/hvp_grad.py#L118
            with torch.no_grad():
                new_estimate = [a + (1 - damp) * b - c / scale
                                for a, b, c in zip(v, last_estimate, this_estim)]

            pbar.update(1)
            if verbose is True:
                new_estimate_norm = new_estimate[0].norm().item()
                last_estimate_norm = last_estimate[0].norm().item()
                estimate_norm_diff = new_estimate_norm - last_estimate_norm
                pbar.set_description(f"{new_estimate_norm:.2f} | {estimate_norm_diff:.2f}")

            cumulative_num_samples += 1
            last_estimate = new_estimate
            if num_samples is not None and i > num_samples:
                break

    # References:
    # https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py#L475
    # Do this for each iteration of estimation
    # Since we use one estimation, we put this at the end
    inverse_hvp = [X / scale for X in last_estimate]

    # Sanity check
    # Note that in parallel settings, we should have `num_samples`
    # whereas in sequential settings we would have `num_samples + 2`.
    # This is caused by some loose stop condition. In parallel settings,
    # We only allocate `num_samples` data to reduce communication overhead.
    # Should probably make this more consistent sometime.
    if cumulative_num_samples not in [num_samples, num_samples + 2]:
        raise ValueError(f"cumulative_num_samples={cumulative_num_samples} f"
                         f"but num_samples={num_samples}: Untested Territory")

    return inverse_hvp


def compute_influences(model: torch.nn.Module, device: torch.device, n_gpu: int,
                       f_loss: Callable,
                       test_dl: torch.utils.data.DataLoader,
                       batch_train_data_loader: torch.utils.data.DataLoader,
                       instance_train_data_loader: torch.utils.data.DataLoader,
                       params_filter: Optional[List[str]] = None,
                       weight_decay: Optional[float] = None,
                       weight_decay_ignores: Optional[List[str]] = None,
                       s_test_damp: float = 3e-5, s_test_scale: float = 1e4,
                       s_test_num_samples: Optional[int] = None, s_test_iterations: int = 1,
                       precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
                       train_indices_to_include: Optional[Union[np.ndarray, List[int]]] = None) \
        -> InfFuncTensors:
    r"""

    :param n_gpu: Number of GPUs not the GPU number
    :param device:
    :param model:
    :param f_loss: Loss function
    :param test_dl: Test DataLoader
    :param batch_train_data_loader: Used to generate the HVP
    :param instance_train_data_loader: Used when calculating the influences
    :param params_filter:
    :param weight_decay:
    :param weight_decay_ignores:
    :param s_test_damp:
    :param s_test_scale:
    :param s_test_num_samples:
    :param s_test_iterations:
    :param precomputed_s_test:
    :param train_indices_to_include:
    :return:
    """

    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must >= 1")

    if weight_decay_ignores is None:
        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    if precomputed_s_test is not None:
        s_test = precomputed_s_test
    else:
        s_test = None
        for _ in range(s_test_iterations):
            _s_test = compute_s_test(model=model, n_gpu=n_gpu, device=device,
                                     f_loss=f_loss, test_dl=test_dl,
                                     train_data_loader=batch_train_data_loader,
                                     params_filter=params_filter, weight_decay=weight_decay,
                                     weight_decay_ignores=weight_decay_ignores, damp=s_test_damp,
                                     scale=s_test_scale, num_samples=s_test_num_samples)

            # Sum the values across runs
            if s_test is None:
                s_test = _s_test
            else:
                s_test = [a + b for a, b in zip(s_test, _s_test)]
        # Do the averaging across multiple random repeats of HVP (i.e., hyperparameter r)
        s_test = [a / s_test_iterations for a in s_test]

    s_test_flat = _flatten(s_test)
    inf_base, inf_sim = [], []
    for index, train_inputs in enumerate(tqdm(instance_train_data_loader, disable=config.QUIET)):

        # Skip indices when a subset is specified to be included
        if (train_indices_to_include is not None) and (index not in train_indices_to_include):
            continue

        xs, ys = train_inputs[0], train_inputs[1]
        grad_z = compute_gradients(n_gpu=n_gpu, device=device, model=model, f_loss=f_loss,
                                   x=xs, y=ys, params_filter=params_filter,
                                   weight_decay=weight_decay,
                                   weight_decay_ignores=weight_decay_ignores,
                                   create_graph=False)

        layer_norm = utils.build_layer_norm(grad_z)  # Must call before flatten to keep structure
        grad_z = _flatten(grad_z)
        inf_base.append(-torch.dot(grad_z, s_test_flat).view([1]))
        # More numerically stable to normalize first to prevent underflows
        inf_sim.append(-torch.dot(grad_z / grad_z.norm(), s_test_flat).view([1]))

    # Construct the final tensors
    tensors = InfFuncTensors()
    tensors.inf_base = torch.cat(inf_base, dim=0).double()
    tensors.inf_sim = torch.cat(inf_sim, dim=0).double()
    tensors.s_test = s_test
    return tensors


def _flatten(vec: Sequence[Tensor]) -> Tensor:
    r""" Flatten the gradient into a vector """
    return torch.cat([flat.detach().view([-1]) for flat in vec], dim=0)


def _normalize_hvp(hvp: Sequence[Tensor]):
    r""" Normalize the Hessian-vector product """
    sqr = sum(torch.sum(torch.square(mat)) for mat in hvp)
    norm = torch.sqrt(sqr)
    for vec in hvp:
        vec /= norm
