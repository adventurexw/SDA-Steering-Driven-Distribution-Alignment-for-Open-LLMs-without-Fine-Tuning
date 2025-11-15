"""
The variables sigma and hyper_k are two introduced hyperparameters. They control, respectively,
the effect of JS divergence on temperature adjustment and the scaling of the SDA influence term.
"""

from typing import List
from functools import partial
import numpy as np
import tqdm

import torch
import torch.nn.functional as F
from transformers import DynamicCache, GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from sda.utils import *

def adjust_temperature_based_on_js(JS_flag, sda_lp, temperature, logits, logits_nc, T_function):
    """
    Adjust temperature based on Jensen-Shannon divergence between two distributions.

    Args:
        JS_flag (bool): Whether to enable JS divergence-based adjustment.
        sda_lp (torch.Tensor): Log-probabilities after SDA modification (not used for JS directly).
        temperature (float): Initial temperature value (scalar).
        logits (torch.Tensor): Logits of the first distribution (N, L, V).
        logits_nc (torch.Tensor): Logits of the second distribution (N, L, V).
        T_function (int): Temperature adjustment strategy.

    Returns:
        torch.Tensor: Temperatures per sample, shape (N,).
    """
    # Ensure temperature is positive
    temperature = max(temperature, 1e-6)
    # JSD max = ln(2) ≈ 0.6931
    sigma = 0.6931

    if JS_flag:
        # Compute softmax probabilities
        p = torch.softmax(logits, dim=-1)
        q = torch.softmax(logits_nc, dim=-1)

        # Mixture distribution
        M = 0.5 * (p + q)

        # KL(p||M) and KL(q||M)
        kl_p_m = torch.sum(p * (torch.log(p + 1e-9) - torch.log(M + 1e-9)), dim=-1)
        kl_q_m = torch.sum(q * (torch.log(q + 1e-9) - torch.log(M + 1e-9)), dim=-1)

        # JS divergence per token, then average over sequence length
        js_div = 0.5 * (kl_p_m + kl_q_m)
        js_div = js_div.mean(dim=-1)  # (N,) per-sample JS divergence
        print("js_div:", js_div)

        # Temperature per sample
        T0 = temperature
        batch_size = js_div.shape[0]
        temperatures = torch.full((batch_size,), T0, dtype=logits.dtype, device=logits.device)

        if T_function == 0:
            temperatures = T0 * (0.5 ** (js_div / (2 * sigma)))
        elif T_function == 1:
            sigma = 0.01
            temperatures = T0 * (0.5 ** (js_div / sigma))
            temperatures = torch.clamp(temperatures, min=0.2)  # avoid extremely low temperatures

        return temperatures
    else:
        # If JS-based adjustment is disabled, return a uniform temperature for all samples
        batch_size = logits.shape[0]
        return torch.full((batch_size,), temperature, dtype=logits.dtype, device=logits.device)



def apply_sda_and_JS(
    logits: torch.Tensor,
    logits_nc: torch.Tensor,
    temperature: float,
    lambdas: torch.Tensor,
    mask: torch.Tensor = None,
    return_probs: bool = False,
    last_token: bool = True,
    JS_flag: bool = True,
    sda_flag: bool = True,
    step_num: int = 0,
    nc_length: int = 100,
    T_function: int = 0,
):
    """
    Apply SDA influence to logits and optionally adjust temperature using JS divergence.

    Args:
        logits (torch.Tensor): Clean logits, shape (N, L, V).
        logits_nc (torch.Tensor): No-context logits, shape (N, L, V).
        temperature (float): Base temperature.
        lambdas (torch.Tensor): Per-sample lambda, shape (N,).
        mask (torch.Tensor, optional): Attention mask (N, L). If None, only last token is used.
        return_probs (bool): If True, return probabilities; otherwise return log-probs.
        last_token (bool): If True, return only the last-token distribution.
        JS_flag (bool): Enable JS-based temperature adjustment.
        sda_flag (bool): Enable SDA influence mix-in.
        step_num (int): Current step (reserved for decay strategies).
        nc_length (int): Total steps for decay (reserved).
        T_function (int): Temperature adjustment strategy id.

    Returns:
        torch.Tensor: (N, V) if last_token else (N, L, V), in prob or log-prob as requested.
    """
    assert len(logits.shape) == len(logits_nc.shape) == 3
    assert len(lambdas.shape) == 1 and len(lambdas) == logits.shape[0]

    # Check for NaNs or Infs in logits
    if torch.isnan(logits).any() or torch.isnan(logits_nc).any():
        raise ValueError("logits or logits_nc contains NaN values.")
    if torch.isinf(logits).any() or torch.isinf(logits_nc).any():
        raise ValueError("logits or logits_nc contains Inf values.")

    if mask is None:
        mask = torch.zeros(logits.shape[:-1], device=logits.device)
        mask[:, -1] = 1  # use last token by default

    # Compute log-probabilities (no temperature at this stage)
    next_lp = F.log_softmax(logits, dim=-1)       # (N, L, V)
    next_lp_nc = F.log_softmax(logits_nc, dim=-1) # (N, L, V)

    # Zero out positions not selected by the mask
    next_lp = next_lp.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0)
    next_lp_nc = next_lp_nc.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0)

    # Influence term
    influence = next_lp - next_lp_nc  # (N, L, V)
    if torch.isnan(influence).any() or torch.isinf(influence).any():
        raise ValueError("influence contains NaN or Inf values.")

    # Example scaling factor for influence
    # hyper_k can be customized or scheduled; using a constant here
    hyper_k = torch.tensor(2.0, dtype=logits.dtype, device=logits.device)

    lambdas = lambdas.to(dtype=logits.dtype, device=logits.device)

    # Control flags (kept consistent with original behavior)
    sda_flag = True
    JS_flag = True

    if sda_flag:
        sda_lp = next_lp + hyper_k * lambdas[:, None, None] * influence  # (N, L, V)
    else:
        sda_lp = next_lp

    # Temperature adjustment returns per-sample temperatures
    temperatures = adjust_temperature_based_on_js(JS_flag, sda_lp, temperature, logits, logits_nc, T_function)
    print("temperatures per sample:", temperatures)  # shape: (N,)

    # Apply per-sample temperature and normalize again
    sda_lp = F.log_softmax(sda_lp / temperatures[:, None, None], dim=-1)  # (N, L, V)

    # Mask out unused positions
    sda_lp = sda_lp.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0)

    output = torch.exp(sda_lp) if return_probs else sda_lp
    output = output.float()

    return output[:, -1] if last_token else output



def get_seqs_logits(input_ids, mask, model, tokenizer, cache=None):
    """
    Generate one token and return the logits for the next step.

    Args:
        input_ids (torch.Tensor): Token ids (batch, seq_len).
        mask (torch.Tensor): Attention mask (batch, seq_len).
        model (PreTrainedModel): HF model.
        tokenizer (PreTrainedTokenizer): Matching tokenizer.
        cache: Optional KV cache.

    Returns:
        torch.Tensor: Logits of the next token, shape (N, V).
    """
    # Ensure tokenizer/model pad ids align and use left padding
    assert model.config.pad_token_id == tokenizer.pad_token_id
    tokenizer.padding_side = 'left'

    with torch.no_grad():
        clean_config = GenerationConfig.from_model_config(model.config)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True if cache is not None else False,
            past_key_values=cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            forced_eos_token_id=None,
            suppress_tokens=[],
            do_sample=False,
            generation_config=clean_config
        )

    if len(output.scores) == 0:
        print("⚠️ No new tokens generated. Possibly EOS reached or input malformed.")
        print("EOS token ID:", tokenizer.eos_token_id)
        print("Input ends with EOS?", input_ids[0, -1] == tokenizer.eos_token_id)
        raise ValueError("generate() did not return any logits.")

    # Validate logits
    logits = output.scores[0]
    assert logits.size(0) == len(input_ids)
    assert not torch.all(logits == -float('inf')), "Logits are all -inf!"
    assert not torch.all(logits == 0), "Logits are all zeros!"
    assert not torch.isnan(logits).any(), "Logits contain NaN values!"

    return logits


