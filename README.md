# Listen-to-your-pulseq
Add a feature to 'pypulseq.Sequence' in order to provide audio preview for grad waveform.

[![pypulseq](https://img.shields.io/badge/-pypulseq-gray?logo=github)](https://github.com/imr-framework/pypulseq)

## Usage

**Example:**

```python
from pypulseq import Sequence
import pypulseq_audio

seq = Sequence()

seq.listen(play_now=True)

duration = seq.duration_update(append_only=True)
```

### **Interface documentation:**


**Sequence.duration_update**

```python
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
```

**Sequence.listen**
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