from resemblyzer.hparams import sampling_rate

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from time import perf_counter, sleep
from sys import stderr


def play_wav(wav, blocking=True):
    try:
        import sounddevice as sd

        # Small bug with sounddevice.play: the audio is cut 0.5 second too early. We pad it to
        # make up for that
        wav = np.concatenate((wav, np.zeros(sampling_rate // 2)))
        sd.play(wav, sampling_rate, blocking=blocking)
    except Exception as e:
        print("Failed to play audio: %s" % repr(e))


def interactive_diarization(times, labels, wav, num_of_speakers):

    fig, ax = plt.subplots()

    (line,) = plt.plot([], [], label="speaker")
    x_crop = 5

    rate = 1 / (times[1] - times[0])
    crop_range = int(np.round(x_crop * rate))
    ticks = np.arange(0, len(times), rate)

    ref_time = perf_counter()

    def init():
        ax.set_ylim(0, num_of_speakers + 2)
        ax.set_yticks(np.arange(0, num_of_speakers + 2, 1))
        ax.set_ylabel("Speaker")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Diarization")
        ax.legend(loc="lower right")
        return (line,)

    def animate(frame):
        crop = (max(frame - crop_range // 2, 0), frame + crop_range // 2)
        ax.set_xlim(frame - crop_range // 2, crop[1])

        crop_ticks = ticks[(crop[0] <= ticks) * (ticks <= crop[1])]
        ax.set_xticks(crop_ticks)
        ax.set_xticklabels(np.round(crop_ticks / rate).astype(np.int))

        line.set_data(range(crop[0], frame + 1), labels[crop[0] : frame + 1] + 1)

        current_time = perf_counter() - ref_time
        if current_time < times[frame]:
            sleep(times[frame] - current_time)
        elif current_time - 0.2 > times[frame]:
            print("Animation is delayed further than 200ms!", file=stderr)
        return (line,)

    ani = FuncAnimation(
        fig,
        animate,
        frames=len(times),
        init_func=init,
        blit=False,
        repeat=False,
        interval=1,
    )
    play_wav(wav, blocking=False)
    plt.show()


def get_time_intervals(times, labels):

    index_of_change = np.where(labels[:-1] != labels[1:])[0]
    index_of_change = np.append(index_of_change, len(times) - 1)

    return np.column_stack((labels, times))[index_of_change]


def log_speaker_diary(time_intervals):
    prev_time = 0, 0
    for speaker_id, time in time_intervals:

        mon, sec = divmod(time, 60)

        print(
            "Speaker {} -> {:02.0f}:{:02.0f}-{:02.0f}:{:02.0f}".format(
                int(speaker_id), *prev_time, mon, sec
            ),
        )
        prev_time = mon, sec