from pypulseq.utils.cumsum import cumsum
from pypulseq import Sequence
from scipy.io.wavfile import write
import numpy as np
import os


def is_jupyter_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:  # Jupyter Notebook 启动特征
            return True
    except:
        pass
    return False


def duration_update(self: Sequence, append_only = True) -> tuple:
    """
    Duration calculation with reduced time cost. Designed for environments where duration is treated as an iteration condition.

    Parameters
    ----------
    self : Sequence
        The sequence object.
    append_only : bool, optional
        If you can ensure that blocks will only be added sequentially and not deleted or inserted, then True. Else False.

    Returns
    ----------
    duration : float
        The total duration of the sequence in seconds.
    """
    if not hasattr(self, '_duration_history'):
        duration = 0
        for block_counter in self.block_events.keys():
            duration += self.block_durations[block_counter]
        self._duration_history = (duration, next(reversed(self.block_events.keys()))) if append_only else (duration, self.block_events.copy())
        return duration
    else:
        duration = self._duration_history[0]
        if append_only:
            start_key = self._duration_history[1]
            keys = list(self.block_events.keys())
            for block_counter in keys[keys.index(start_key)+1:]:
                duration += self.block_durations[block_counter]
        else:
            for block_counter in set(self.block_events.keys()).difference(self._duration_history[1].keys()):
                duration += self.block_durations[block_counter]
            for block_counter in set(self._duration_history[1].keys()).difference(self.block_events.keys()):
                duration -= self.block_durations[block_counter]
        self._duration_history = (duration, next(reversed(self.block_events.keys()))) if append_only else (duration, self.block_events.copy())
        return duration


def listen(
    self: Sequence,
    speaker: callable = None,
    save_path: str = None,
    time_range=(0, np.inf),
    time_disp: str = 's',
    grad_disp: str = 'kHz/m',
    play_now: bool = True,
    rate: int = 44100,
) -> None:
    """
    Listen to the waveform of the sequence.

    Parameters
    ----------
    self : Sequence
        The sequence object.
    speaker : callable, optional
        The speaker function to play the audio. If None, use IPython.display.Audio.
    save_path : str, optional
        The path to save the waveform as a .wav file.
    time_range : tuple, optional
        The time range to listen to, in seconds. Default is (0, np.inf).
    time_disp : str, optional
        The time unit for the waveform. Default is 's'.
    grad_disp : str, optional
        The gradient unit for the waveform. Default is 'kHz/m'.
    play_now : bool, optional
        Whether to play the audio immediately. Default is True.
    rate : int, optional
        The sample rate for the audio. Default is 44100.
    
    Returns
    -------
    waveform : np.ndarray
        The waveform of the sequence.

    """

    if speaker is None:
        if is_jupyter_notebook():
            from IPython.display import Audio, display
            speaker = Audio

    # waveform = self.waveforms_and_times(time_range=time_range)

    valid_time_units = ['s', 'ms', 'us']
    valid_grad_units = ['kHz/m', 'mT/m']
    if not all(isinstance(x, (int, float)) for x in time_range) or len(time_range) != 2:
        raise ValueError('Invalid time range')
    if time_disp not in valid_time_units:
        raise ValueError('Unsupported time unit')

    t_factor_list = [1, 1e3, 1e6]
    t_factor = t_factor_list[valid_time_units.index(time_disp)]

    g_factor_list = [1e-3, 1e3 / self.system.gamma]
    g_factor = g_factor_list[valid_grad_units.index(grad_disp)]

    
    buffer = []
    t0 = 0
    t00 = None
    for block_counter in self.block_events:
        block = self.get_block(block_counter)
        is_valid = time_range[0] <= t0 + self.block_durations[block_counter] and t0 <= time_range[1]
        if is_valid:
            if t00 is None:
                t00 = t0
            grad_channels = ['gx', 'gy', 'gz']
            for x in range(len(grad_channels)):  # Gradients
                if getattr(block, grad_channels[x], None) is not None:
                    grad = getattr(block, grad_channels[x])
                    if grad.type == 'grad':
                        time = (grad.delay + np.array([0, *grad.tt, grad.shape_dur]))
                        waveform = g_factor * np.array((grad.first, *grad.waveform, grad.last))
                    else:
                        time = np.array(
                            cumsum(
                                0,
                                grad.delay,
                                grad.rise_time,
                                grad.flat_time,
                                grad.fall_time,
                            ))
                        waveform = g_factor * grad.amplitude * np.array([0, 0, 1, 1, 0])
                    buffer.append((time + t0, waveform))
        t0 += self.block_durations[block_counter]

    time = np.concatenate([x[0] for x in buffer])
    waveform = np.concatenate([x[1] for x in buffer])
    time_new = np.arange(0, int(np.max(time)*rate), dtype=np.float64) * 1 / rate  # New time axis
    total_waveform = np.interp(time_new, time, waveform)

    if play_now:
        if is_jupyter_notebook():
            display(speaker(total_waveform, rate=rate))

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        write(os.path.join(save_path, 'seq.wav'), rate=rate, data=total_waveform)

    return total_waveform


def _listentoyourpulseq_patch():
    if not hasattr(Sequence, "listen"):
        Sequence.listen = listen
        Sequence.duration_update = duration_update
    else:
        print("listen function already exists in Sequence class.")