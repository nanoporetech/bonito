import numpy as np

from remora.data_chunks import RemoraRead
from remora.inference import call_read_mods


def mods_tags_to_str(mods_tags):
    return [
        f"MM:Z:{mods_tags[0]}",
        f"ML:B:C,{','.join(map(str, mods_tags[1]))}"
    ]


def call_mods(mods_model, read, read_attrs):
    if len(read_attrs['sequence']) == 0:
        return read_attrs
    remora_model, remora_metadata = mods_model
    seq_to_sig_map = np.empty(
        len(read_attrs['sequence']) + 1, dtype=np.int32
    )
    seq_to_sig_map[-1] = read.signal.shape[0]
    seq_to_sig_map[:-1] = read_attrs['seq_to_sig_map']
    remora_read = RemoraRead(
        read.signal,
        seq_to_sig_map,
        str_seq=read_attrs['sequence'].upper(),
    )
    read_attrs['mods'] = call_read_mods(
        remora_read,
        remora_model,
        remora_metadata,
        return_mm_ml_tags=True,
    )
    return read_attrs
