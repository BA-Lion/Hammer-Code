import pytest

from hammer_code.cli import main


def test_help_requires_no_config_or_key() -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
