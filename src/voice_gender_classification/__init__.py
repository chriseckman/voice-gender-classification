"""Package level utilities for :mod:`voice_gender_classification`."""

__version__ = "0.1.0"


class _LazyPipeline:
    """Lazy loader for :class:`GenderClassificationPipeline`.

    Importing :mod:`voice_gender_classification` in environments where
    heavy dependencies such as ``speechbrain`` and ``torchaudio`` are not
    available would fail if we eagerly imported the pipeline class.  The
    tests monkeypatch parts of these dependencies *after* importing this
    package, so we defer the actual import of :mod:`pipeline` until the
    class is instantiated or a classmethod is accessed.
    """

    def __new__(cls, *args, **kwargs):
        from .pipeline import GenderClassificationPipeline as _Real

        return _Real(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from .pipeline import GenderClassificationPipeline as _Real

        return _Real.from_pretrained(*args, **kwargs)


GenderClassificationPipeline = _LazyPipeline
__all__ = ["GenderClassificationPipeline"]
