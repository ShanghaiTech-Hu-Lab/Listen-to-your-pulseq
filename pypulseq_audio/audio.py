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

def listen(
    self: Sequence,
    speaker: callable = None,
    save_path: str = None,
    time_range=(0, np.inf),
    time_disp: str = 's',
    grad_disp: str = 'kHz/m',
    play_now: bool = True,
    rate: int = 44100,
) -> np.ndarray:
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

    valid_time_units = ['s', 'ms', 'us']
    valid_grad_units = ['kHz/m', 'mT/m']
    if not all(isinstance(x, (int, float)) for x in time_range) or len(time_range) != 2:
        raise ValueError('Invalid time range')
    if time_disp not in valid_time_units:
        raise ValueError('Unsupported time unit')
    
    rate = int(1/self.system.grad_raster_time)

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
                        time = (grad.delay + np.array([0, *grad.tt, grad.shape_dur]) + t0)
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
                    time_new =( grad.delay + np.linspace(0, grad.shape_dur, int(grad.shape_dur * rate), endpoint=False) + t0)
                    buffer.append((time_new, time, waveform))
        t0 += self.block_durations[block_counter]

    time_new = np.concatenate([x[0] for x in buffer])
    time = np.concatenate([x[1] for x in buffer])
    waveform = np.concatenate([x[2] for x in buffer])
    total_waveform = np.interp(time_new, time, waveform)

    print(total_waveform.shape, time_new.shape, time.shape, waveform.shape)

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
    else:
        print("listen function already exists in Sequence class.")