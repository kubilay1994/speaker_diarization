from resemblyzer import preprocess_wav, VoiceEncoder
from resemblyzer.hparams import sampling_rate
import numpy as np

from spectralcluster import SpectralClusterer
from pathlib import Path

import argparse
from utils import log_speaker_diary, get_time_intervals, interactive_diarization


parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True, type=str)
parser.add_argument("--rate", "-r", type=float, default=1.3)

parser.add_argument(
    "--num",
    help="Num of speakers(optional)",
    default=None,
    type=int,
)

parser.add_argument("--interactive", dest="interactive", action="store_true")
parser.add_argument("--no-interactive", dest="interactive", action="store_false")
parser.set_defaults(interactive=True)

args = parser.parse_args()

audio_path = Path(args.audio)
wav = preprocess_wav(audio_path)

encoder = VoiceEncoder() if args.rate <= 4 else VoiceEncoder("cpu")

# encoder = VoiceEncoder("cpu")
_, cond_emd, wav_splits = encoder.embed_utterance(
    wav,
    return_partials=True,
    rate=args.rate,
)

clusterer = SpectralClusterer(
    min_clusters=args.num,
    p_percentile=0.91,
)


labels = clusterer.predict(cond_emd)
times = np.array([(s.start + s.stop) / 2 / sampling_rate for s in wav_splits])


if args.interactive:
    interactive_diarization(times, labels, wav, labels.max() + 1)
else:
    time_intervals = get_time_intervals(times, labels)
    log_speaker_diary(time_intervals)
