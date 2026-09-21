from tyrannis.processor import Joblib


def test_joblib_accepts_documented_serialized_backend() -> None:
    Joblib(joblib_backend="serialized")
