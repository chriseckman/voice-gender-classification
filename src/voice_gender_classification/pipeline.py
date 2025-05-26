from typing import Any, Dict, List, Union

from soundfile import read as sf_read
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor, tensor, zeros


class GenderClassificationPipeline:
    """Very small gender classification pipeline used for testing."""

    def __init__(self, svm_model: Any, scaler: Any, device: str = "cpu"):
        self.device = device
        self.model = EncoderClassifier.from_hparams(source="dummy", run_opts={"device": device})
        self.svm_model = svm_model
        self.scaler = scaler
        self.labels = ["female", "male"]
        self.feature_names = [f"{i}_speechbrain_embedding" for i in range(192)]

    def _load_audio(self, path: str) -> List[float]:
        waveform, _sr = sf_read(path)
        return waveform if isinstance(waveform, list) else list(waveform)

    def preprocess(self, audio_input: Union[str, List[str], List[float], Tensor]) -> Dict[str, Tensor]:
        if isinstance(audio_input, list) and all(isinstance(i, str) for i in audio_input):
            waves = [self._load_audio(p) for p in audio_input]
        elif isinstance(audio_input, str):
            waves = [self._load_audio(audio_input)]
        elif isinstance(audio_input, Tensor):
            waves = [audio_input]
        elif hasattr(audio_input, "__iter__"):
            waves = [list(audio_input)]
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        inputs = Tensor(waves)
        wav_lens = Tensor([1.0 for _ in waves])
        return {"inputs": inputs, "wav_lens": wav_lens}

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        return self.model.encode_batch(inputs["inputs"], inputs["wav_lens"])

    def postprocess(self, model_outputs: Tensor) -> List[str]:
        embeddings = [emb[0] for emb in model_outputs]
        scaled = self.scaler.transform(embeddings)
        preds = self.svm_model.predict(scaled)
        return [self.labels[p] for p in preds]

    def __call__(self, audio_input: Union[str, List[str], List[float], Tensor]) -> List[str]:
        inputs = self.preprocess(audio_input)
        outputs = self.forward(inputs)
        return self.postprocess(outputs)

