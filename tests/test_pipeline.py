import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import torch
import soundfile as sf
import pytest

from voice_gender_classification import GenderClassificationPipeline

class DummyEncoder:
    def __init__(self, embedding_dim=192):
        self.embedding = torch.zeros(1, 1, embedding_dim)

    def encode_batch(self, inputs, wav_lens):
        batch_size = inputs.shape[0]
        return self.embedding.repeat(batch_size, 1, 1)

def dummy_from_hparams(*args, **kwargs):
    return DummyEncoder()

class DummyScaler:
    def transform(self, X):
        return X

class DummySVM:
    def predict(self, X):
        return [0] * len(X)

@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(
        'speechbrain.inference.speaker.EncoderClassifier.from_hparams',
        dummy_from_hparams
    )
    return GenderClassificationPipeline(svm_model=DummySVM(), scaler=DummyScaler())


def create_wav(path, duration=1.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, wave, sr)


def test_single_file_prediction(pipeline, tmp_path):
    audio_path = tmp_path / 'audio.wav'
    create_wav(audio_path)
    result = pipeline(str(audio_path))
    assert result == ['female']


def test_batch_prediction(pipeline, tmp_path):
    audio1 = tmp_path / 'a1.wav'
    audio2 = tmp_path / 'a2.wav'
    create_wav(audio1)
    create_wav(audio2)
    result = pipeline([str(audio1), str(audio2)])
    assert result == ['female', 'female']


def test_nonexistent_file(pipeline):
    with pytest.raises(Exception):
        pipeline('no_such_file.wav')


def test_corrupted_file(pipeline, tmp_path):
    bad_path = tmp_path / 'bad.wav'
    bad_path.write_text('not audio')
    with pytest.raises(Exception):
        pipeline(str(bad_path))


def test_invalid_input_type(pipeline):
    with pytest.raises(ValueError):
        pipeline(123)


def test_numpy_input(pipeline):
    arr = np.zeros(16000, dtype=np.float32)
    result = pipeline(arr)
    assert result == ['female']


def test_tensor_input(pipeline):
    arr = torch.zeros(16000)
    result = pipeline(arr)
    assert result == ['female']


def test_short_audio(pipeline, tmp_path):
    audio_path = tmp_path / 'short.wav'
    create_wav(audio_path, duration=0.01)
    result = pipeline(str(audio_path))
    assert result == ['female']