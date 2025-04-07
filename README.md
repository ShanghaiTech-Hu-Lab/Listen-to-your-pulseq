# Listen-to-your-pulseq
Add a feature to 'pypulseq.Sequence' in order to provide audio preview for grad waveform.

[![pypulseq](https://img.shields.io/badge/-pypulseq-gray?logo=github)](https://github.com/imr-framework/pypulseq)

## Usage

**Example:**

```python
from pypulseq import Sequence
import pypulseq_audio

seq = Sequence()
seq.listen()
```

**Interface documentation:**

```python
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
```