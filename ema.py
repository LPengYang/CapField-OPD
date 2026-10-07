"""Exponential moving average of trainable parameters (save-time only).

The EMA shadow weights never take part in the training forward/backward; they
are swapped in only while writing `step_N_ema/` checkpoints. Decay follows the
same warmup ramp as DiffusionOPD: decay_t = min((1 + step) / (10 + step), decay).
"""

from collections.abc import Iterable

import torch


class EMAModuleWrapper:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9,
        update_step_interval: int = 8,
        device: torch.device | None = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        self.temp_stored_parameters = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step) -> float:
        return min((1 + optimization_step) / (10 + optimization_step), self.decay)

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step):
        parameters = list(parameters)
        one_minus_decay = 1 - self.get_current_decay(optimization_step)
        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
                    else:
                        diff = parameter.detach().to(ema_parameter.device)
                        diff.sub_(ema_parameter).mul_(one_minus_decay)
                        ema_parameter.add_(diff)
                        del diff

    @torch.no_grad()
    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True):
        """Swap EMA weights into the model (call `copy_temp_to` afterwards)."""
        parameters = list(parameters)
        if store_temp:
            self.temp_stored_parameters = [p.detach().clone() for p in parameters]
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    @torch.no_grad()
    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]):
        """Restore the training weights saved by `copy_ema_to`."""
        parameters = list(parameters)
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            parameter.data.copy_(temp_parameter.to(parameter.device))
        self.temp_stored_parameters = None
