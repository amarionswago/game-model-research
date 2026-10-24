"""Bounded exact-byte tensor hashing candidate; no scan may be skipped.

Partitioning a fixed byte stream does not change SHA256. This implementation
keeps the original detached logical-contiguous ordering, including for views,
but bounds transfers by bytes and passes the host buffer directly to hashlib.
No GPU throughput or provider admission is established by this local module.
"""
import hashlib

import torch

COPY_BYTES = 8 * 1024 * 1024


def update_tensor_hash(hasher, tensor, *, copy_bytes=COPY_BYTES):
    if tensor.layout != torch.strided or tensor.device.type == 'meta':
        raise ValueError('Unhashable tensor layout/device')
    if type(copy_bytes) is not int or not tensor.element_size() <= copy_bytes <= COPY_BYTES:
        raise ValueError('Copy allowance must fit one element and stay within 8 MiB')
    flat = tensor.detach().reshape(-1)
    elements = copy_bytes // tensor.element_size()
    for chunk in flat.split(elements):
        host = chunk.cpu().contiguous()
        array = host.view(torch.uint8).numpy()
        hasher.update(memoryview(array))
        # Release the host buffer before starting the next device transfer.
        del array, host


def tensor_identity(tensor):
    hasher = hashlib.sha256()
    update_tensor_hash(hasher, tensor)
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': hasher.hexdigest()}
