import sys
import logging
import numpy as np

from remora import log
from remora.model_util import load_model
from remora.data_chunks import RemoraRead
from remora.inference import call_read_mods

class CustomFormatter(logging.Formatter):
    err_fmt = "> error (remora): %(msg)s"
    warn_fmt = "> warning (remora): %(msg)s"
    info_fmt = "> %(msg)s"

    def __init__(self, fmt="> %(message)s"):
        super().__init__(fmt=fmt, style="%")

    def format(self, record):
        format_orig = self._fmt
        if record.levelno == logging.INFO:
            self._style._fmt = self.info_fmt
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = self.fmt
        result = logging.Formatter.format(self, record)
        self._fmt = format_orig
        return result

log.CONSOLE.setLevel(logging.WARNING)
log.CONSOLE.setFormatter(CustomFormatter())


def load_mods_model(mod_bases, bc_model_str, model_path):
    if mod_bases is not None:
        try:
            bc_model_type, model_version = bc_model_str.split('@')
            bc_model_type_attrs = bc_model_type.split('_')
            pore = '_'.join(bc_model_type_attrs[:-1])
            bc_model_subtype = bc_model_type_attrs[-1]
        except:
            sys.stderr.write(
                f"Could not parse basecall model directory ({bc_model_str}) "
                "for automatic modified base model loading"
            )
            sys.exit(1)
        return load_model(
            pore=pore,
            basecall_model_type=bc_model_subtype,
            basecall_model_version=model_version,
            modified_bases=mod_bases,
            quiet=True,
        )
    return load_model(model_path, quiet=True)


def mods_tags_to_str(mods_tags):
    return [
        f"MM:Z:{mods_tags[0]}",
        f"ML:B:C,{','.join(map(str, mods_tags[1]))}"
    ]


def call_mods(mods_model, read, read_attrs):
    if len(read_attrs['sequence']) == 0:
        return read_attrs
    remora_model, remora_metadata = mods_model
    # convert signal move table to remora read format
    seq_to_sig_map = np.empty(
        len(read_attrs['sequence']) + 1, dtype=np.int32
    )
    seq_to_sig_map[-1] = read.signal.shape[0]
    seq_to_sig_map[:-1] = np.where(read_attrs['sig_move'])[0]
    remora_read = RemoraRead(
        read.signal,
        seq_to_sig_map,
        str_seq=read_attrs['sequence'].upper(),
    )
    read_attrs['mods'] = mods_tags_to_str(
        call_read_mods(
            remora_read,
            remora_model,
            remora_metadata,
            return_mm_ml_tags=True,
        )
    )
    return read_attrs
