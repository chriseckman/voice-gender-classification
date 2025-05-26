class EncoderClassifier:
    @classmethod
    def from_hparams(cls, *args, **kwargs):
        raise NotImplementedError("Dummy stub - replace in tests")

    def encode_batch(self, inputs, wav_lens):
        raise NotImplementedError
