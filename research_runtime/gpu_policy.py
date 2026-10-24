"""Serial inference with selected members and unchanged repaired arithmetic."""
import gc
import math
import time
import sys

import numpy as np
import torch
from transformers import DynamicCache

HybridCache = DynamicCache


class ResponseDeadline(TimeoutError):
    pass


def combine(logs, hidden, gate):
    if gate is None:
        return logs.logsumexp(0)-math.log(2), torch.full(logs.shape[1:2], .5, device=logs.device)
    a = gate(hidden)
    weights = torch.stack([torch.nn.functional.logsigmoid(a), torch.nn.functional.logsigmoid(-a)])
    return (logs+weights[..., None]).logsumexp(0), a.sigmoid()


def synchronize():
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


def memory_snapshot():
    return [dict(device=i, allocated_bytes=torch.cuda.memory_allocated(i),
                 reserved_bytes=torch.cuda.memory_reserved(i),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(i),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(i))
            for i in range(torch.cuda.device_count())]


def cache_snapshot(caches):
    result = {}
    seen = set()
    for member, cache in caches.items():
        tensors = [value for layer in cache.layers for value in (layer.keys, layer.values)
                   if isinstance(value, torch.Tensor)]
        stores = {(str(value.device), value.untyped_storage().data_ptr()) for value in tensors}
        if seen & stores:
            raise RuntimeError('Members share cache storage')
        seen |= stores
        result[str(member)] = dict(bytes=sum(value.numel()*value.element_size() for value in tensors),
                                   tensors=len(tensors),
                                   sequence_tokens=cache.get_seq_length(),
                                   dtypes=sorted({str(value.dtype) for value in tensors}))
    return result


class Population:
    def __init__(self, runtime, gate, *, members=(0, 1)):
        self.members = tuple(members)
        if self.members not in ((0,), (1,), (0, 1)):
            raise ValueError('Population requires member 0, member 1, or both in order')
        if len(self.members) == 1 and gate is not None:
            raise ValueError('Solo member inference cannot use a gate')
        self.runtime, self.gate = runtime, gate
        self.checkpoint_generation = getattr(runtime, 'checkpoint_generation', 0)

    def _forward(self, member, tokens, start, cache):
        r = self.runtime
        if getattr(r, 'checkpoint_generation', 0) != self.checkpoint_generation:
            raise RuntimeError('Population belongs to an earlier checkpoint activation')
        r.model.set_adapter('member'+str(member))
        ids = torch.tensor([tokens], device=r.input_device, dtype=torch.long)
        pos = torch.arange(start, start+len(tokens), device=ids.device)[None]
        mask = torch.ones((1, start+len(tokens)), device=ids.device, dtype=torch.long)
        return r.backbone(input_ids=ids, attention_mask=mask, position_ids=pos,
                          past_key_values=cache, use_cache=True).last_hidden_state[0]

    def _policy(self, logs, hidden):
        if len(self.members) == 1:
            return logs[0], None
        return combine(logs, hidden, self.gate)

    @torch.no_grad()
    def forced(self, sequence, positions, *, chunk_size=64, deadline):
        if not sequence or not positions or chunk_size <= 0:
            raise ValueError('A nonempty sequence, positions, and positive chunk size are required')
        if any(p < 0 or p >= len(sequence) for p in positions) or len(set(positions)) != len(positions):
            raise ValueError('Positions must be unique and inside the sequence')
        caches = {m: HybridCache(config=self.runtime.model.config) for m in self.members}
        saved = {}
        try:
            for lo in range(0, len(sequence), chunk_size):
                hi = min(lo+chunk_size, len(sequence))
                choose = [p for p in positions if lo <= p < hi]
                values = []
                for member in self.members:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('GPU numerical check deadline')
                    hidden = self._forward(member, sequence[lo:hi], lo, caches[member])
                    if choose:
                        selected = hidden[[p-lo for p in choose]]
                        values.append((self.runtime.head(selected).float().log_softmax(-1), selected))
                if choose:
                    logs = torch.stack([v[0] for v in values])
                    hidden = torch.stack([v[1] for v in values])
                    policy, alpha = self._policy(logs, hidden)
                    for i, p in enumerate(choose):
                        saved[p] = (logs[:, i].cpu().numpy(), hidden[:, i].cpu().numpy(),
                                    policy[i].cpu().numpy(), None if alpha is None else alpha[i].cpu().numpy())
            cache_snapshot(caches)
            return (np.stack([saved[p][0] for p in positions], axis=1),
                    np.stack([saved[p][1] for p in positions], axis=1),
                    np.stack([saved[p][2] for p in positions]),
                    None if len(self.members) == 1 else np.stack([saved[p][3] for p in positions]))
        finally:
            del caches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(self, prompt, *, eos_ids, max_new_tokens, deadline, progress=lambda **kw: None):
        if not prompt or max_new_tokens <= 0:
            raise ValueError('A nonempty prompt and positive token cap are required')
        synchronize()
        started = time.monotonic()
        last = started
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        memory_before = memory_snapshot()
        caches = {m: HybridCache(config=self.runtime.model.config) for m in self.members}
        tokens, logps, weights, values = [], [], [], []
        forward_calls = {str(m): 0 for m in self.members}
        stop = 'max_new_tokens'
        prefill_done = None
        first_token = None
        answer = None
        try:
            for lo in range(0, len(prompt), 64):
                hi = min(lo+64, len(prompt))
                values = []
                for member in self.members:
                    if time.monotonic() >= deadline:
                        raise ResponseDeadline('Response deadline')
                    hidden = self._forward(member, prompt[lo:hi], lo, caches[member])[-1:]
                    forward_calls[str(member)] += 1
                    if hi == len(prompt):
                        values.append((self.runtime.head(hidden).float().log_softmax(-1), hidden))
            synchronize()
            prefill_done = time.monotonic()
            for index in range(max_new_tokens):
                if time.monotonic() >= deadline:
                    raise ResponseDeadline('Response deadline')
                logs = torch.stack([v[0] for v in values])
                hidden = torch.stack([v[1] for v in values])
                policy, alpha = self._policy(logs, hidden)
                if not torch.isfinite(policy).all():
                    raise FloatingPointError('Nonfinite generation policy')
                token = int(policy.argmax(-1)[0])
                tokens.append(token)
                logps.append(logs[:, 0, token].cpu().tolist())
                if alpha is not None:
                    weights.append(float(alpha[0]))
                if first_token is None:
                    first_token = time.monotonic()
                if token in eos_ids:
                    stop = 'eos'
                    break
                if time.monotonic()-last >= 30:
                    progress(tokens=len(tokens), seconds=time.monotonic()-started)
                    last = time.monotonic()
                if index+1 == max_new_tokens:
                    break
                values = []
                for member in self.members:
                    if time.monotonic() >= deadline:
                        raise ResponseDeadline('Response deadline')
                    hidden = self._forward(member, [token], len(prompt)+index, caches[member])[-1:]
                    forward_calls[str(member)] += 1
                    values.append((self.runtime.head(hidden).float().log_softmax(-1), hidden))
        except ResponseDeadline:
            stop = 'resource_deadline'
        finally:
            synchronize()
            ended = time.monotonic()
            cache_usage = cache_snapshot(caches)
            memory_after = memory_snapshot()
            answer = dict(tokens=tokens, stop=stop, seconds=ended-started, member_ids=list(self.members),
                          member_logps=logps, member0_weights=weights if len(self.members) == 2 else None,
                          prefill_seconds=(ended if prefill_done is None else prefill_done)-started,
                          prefill_complete=prefill_done is not None,
                          decode_seconds=None if prefill_done is None else ended-prefill_done,
                          first_token_seconds=None if first_token is None else first_token-started,
                          decode_tokens=max(0, len(tokens)-1), prompt_tokens=len(prompt),
                          forward_calls=forward_calls, cache=cache_usage,
                          gpu_memory_before=memory_before, gpu_memory_after=memory_after)
            del caches, values
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            answer['cleanup_seconds'] = time.monotonic()-ended
            answer['wall_seconds'] = time.monotonic()-started
            failure = sys.exc_info()[1]
            if failure is not None:
                failure.baseline_partial_answer = answer
        return answer
