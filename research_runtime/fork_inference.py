"""Single revision-fork inference using the unchanged pilot policy operations."""
import torch

from gpu_policy import Population


class RevisionPopulation(Population):
    """An explicitly installed fork, with the original policies kept frozen.

    The caller must increment runtime.checkpoint_generation after any parameter
    update or restore. The inherited forward rejects an earlier activation.
    This component does not create an adapter, modify weights, or train it.
    """

    def __init__(self, runtime, *, fork_member, expected_parameter_names,
                 expected_checkpoint_generation, frozen_modules=()):
        if type(fork_member) is not int or fork_member < 2:
            raise ValueError('Revision inference requires a separate fork numbered at least 2')
        if type(expected_checkpoint_generation) is not int or expected_checkpoint_generation < 0:
            raise ValueError('An explicit nonnegative activation generation is required')
        actual_generation = getattr(runtime, 'checkpoint_generation', None)
        if type(actual_generation) is not int or actual_generation != expected_checkpoint_generation:
            raise ValueError('Runtime activation generation differs')
        adapter = 'member' + str(fork_member)
        if adapter not in runtime.model.peft_config:
            raise ValueError('Revision fork is not installed')
        names = tuple(expected_parameter_names)
        if not names or len(set(names)) != len(names):
            raise ValueError('Explicit unique fork parameter names are required')
        parameters = dict(runtime.model.named_parameters())
        actual = {name for name in parameters if '.lora_' in name and '.' + adapter + '.' in name}
        if set(names) != actual:
            raise ValueError('Installed fork parameter membership differs')
        if any(parameters[name].dtype != torch.float32 or parameters[name].ndim != 2 for name in names):
            raise ValueError('Revision fork must contain FP32 LoRA matrices')
        if any(parameter.grad is not None for parameter in parameters.values()):
            raise ValueError('Inference requires cleared parameter gradients')
        if any(parameter.requires_grad for name, parameter in parameters.items() if name not in actual):
            raise ValueError('Original members and base must remain frozen')
        if any(parameter.requires_grad or parameter.grad is not None
               for module in frozen_modules for parameter in module.parameters()):
            raise ValueError('External policy or gate must remain frozen')
        # The parent constructor deliberately admits only the original two
        # members. Set its four state fields after validating this distinct
        # single-fork contract; inherit all numerical and generation methods.
        self.members = (fork_member,)
        self.runtime = runtime
        self.gate = None
        self.checkpoint_generation = expected_checkpoint_generation
