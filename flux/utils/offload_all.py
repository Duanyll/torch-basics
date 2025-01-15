import torch
from accelerate.hooks import (
    ModelHook,
    clear_device_cache,
    send_to_device,
    add_hook_to_module,
    remove_hook_from_module,
)
from diffusers.utils import logging

logger = logging.get_logger(__name__)

class OffloadAllHook(ModelHook):
    r"""
    A module hook that try to offload all the given modules to another device before the forward pass. This is more useful
    than the `CpuOffloadHook` when you want to call the components of a pipeline in arbitrary order, as the `CpuOffloadHook`
    requires the components to be called in a specific order.
    """

    def __init__(self, execution_device, offload_device, targets):
        r"""
        Args:
            execution_device (str):
                The device to execute the forward pass on.
            offload_device (str):
                The device to offload the modules to before the forward pass.
            targets (List[torch.nn.Module]):
                The modules to offload.
        """
        super().__init__()
        self.execution_device = torch.device(execution_device)
        self.offload_device = torch.device(offload_device)
        self.targets = targets

    def init_hook(self, module):
        logger.info(
            f"Initalizing OffloadAllHook for {module.__class__.__name__}, moving to {self.offload_device}"
        )
        return module.to(self.offload_device)

    def pre_forward(self, module, *args, **kwargs):
        has_offloaded = False
        for m in self.targets:
            if m is module:
                continue
            if m.device == self.execution_device:
                logger.info(f"Offloading {m.__class__.__name__} to {self.offload_device}")
                if hasattr(m, "lp_cache"):
                    logger.info(f"Clearing lp_cache for {m.__class__.__name__}")
                    m.lp_cache.clear()
                m.to(self.offload_device)
                has_offloaded = True
        if has_offloaded:
            clear_device_cache()
            logger.info(
                f"Flushed device cache, current allocated memory: {torch.cuda.memory_allocated() / 1e9:.4f} GB"
            )
        if module.device != self.execution_device:
            logger.info(f"Loading {module.__class__.__name__} to {self.execution_device}")
            module.to(self.execution_device)
            logger.info(
                f"Done loading, current allocated memory: {torch.cuda.memory_allocated() / 1e9:.4f} GB"
            )
        return send_to_device(args, self.execution_device), send_to_device(
            kwargs, self.execution_device
        )


class UserOffloadAllHook:
    def __init__(self, model, hook):
        self.model = model
        self.hook = hook

    def offload(self):
        self.hook.init_hook(self.model)

    def remove(self):
        remove_hook_from_module(self.model)


def apply_offload_all_hook(pipe, execution_device, offload_device, submodules):
    r"""
    Install the `OffloadAllHook` to the given submodules of a pipeline. This hook will offload the given submodules to
    another device before the forward pass. This is useful when you want to call the components of a pipeline in arbitrary
    order, as the `CpuOffloadHook` requires the components to be called in a specific order.
    Args:
        pipe (diffusers.models.pipeline.Pipeline):
            The pipeline to install the hook to.
        execution_device (torch.device):
            The device to execute the forward pass on. (e.g., "cuda:0")
        offload_device (torch.device):
            The device to offload the modules to before the forward pass. (e.g., "cpu")
        submodules (List[str]):
            The names of the submodules to offload.
    """
    modules = [pipe.components[m] for m in submodules]
    pipe._all_hooks = []
    for module in modules:
        targets = [m for m in modules if m != module]
        hook = OffloadAllHook(execution_device, offload_device, targets)
        add_hook_to_module(module, hook, append=True)
        user_hook = UserOffloadAllHook(module, hook)
        pipe._all_hooks.append(user_hook)
