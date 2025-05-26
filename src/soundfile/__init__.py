import wave
import struct
from typing import List, Tuple


def write(path: str, data: List[float], samplerate: int):
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        frames = b''.join(struct.pack('<h', int(max(-1.0, min(1.0, x)) * 32767)) for x in data)
        wf.writeframes(frames)


def read(path: str) -> Tuple[List[float], int]:
    with wave.open(str(path), 'rb') as wf:
        samplerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        samples = struct.unpack('<' + 'h' * wf.getnframes(), frames)
        data = [s / 32767.0 for s in samples]
    return data, samplerate
