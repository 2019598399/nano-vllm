import torch.distributed as dist


_TP_GROUP = None


def set_tp_group(group):
    global _TP_GROUP
    _TP_GROUP = group


def get_tp_group():
    return _TP_GROUP


def get_tp_rank():
    return dist.get_rank(group=_TP_GROUP)


def get_tp_world_size():
    return dist.get_world_size(group=_TP_GROUP)


def tp_all_reduce(tensor):
    dist.all_reduce(tensor, group=_TP_GROUP)


def tp_gather(tensor, gather_list, dst=0):
    dist.gather(tensor, gather_list, dst=dst, group=_TP_GROUP)
