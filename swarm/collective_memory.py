"""Keep worker-wide allocator peaks across inherited per-generation resets."""
from contextlib import contextmanager
import resource
import torch
from swarm.collective_contract import atomic


@contextmanager
def whole_memory(root):
    original=torch.cuda.reset_peak_memory_stats
    report=dict(devices={},sample_errors=[],scope='whole worker including parent admission; allocator peaks, not minimum deployment memory')
    def sample():
        report['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        try:
            for i in range(torch.cuda.device_count()):
                free,total=torch.cuda.mem_get_info(i)
                row=report['devices'].setdefault(str(i),dict(name=torch.cuda.get_device_name(i),total_bytes=total,
                    minimum_observed_free_bytes=free,peak_allocated_bytes=0,peak_reserved_bytes=0))
                row['minimum_observed_free_bytes']=min(row['minimum_observed_free_bytes'],free)
                row['peak_allocated_bytes']=max(row['peak_allocated_bytes'],torch.cuda.max_memory_allocated(i))
                row['peak_reserved_bytes']=max(row['peak_reserved_bytes'],torch.cuda.max_memory_reserved(i))
        except Exception as error:report['sample_errors'].append(dict(type=type(error).__name__,message=str(error)))
        atomic(root/'whole-worker-memory.json',report)
    def reset(device=None):
        sample()
        return original(device)
    torch.cuda.reset_peak_memory_stats=reset
    try:
        sample();yield report
    finally:
        torch.cuda.reset_peak_memory_stats=original
        sample()
