import pkcore


def test_version_is_exposed() -> None:
    assert pkcore.__version__ == "0.1.0"
