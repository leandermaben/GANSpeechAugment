"""
Implementing functions to collect valid audio files for creating dataset.
"""

import os
import os.path

AUDIO_EXTENSIONS = [
    '.wav'
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_dataset_audio(dir, max_dataset_size=float("inf")):
    audios = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_audio_file(fname):
                path = os.path.join(root, fname)
                audios.append(path)
    return audios[:min(max_dataset_size, len(audios))]

