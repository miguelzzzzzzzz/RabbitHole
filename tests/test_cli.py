from __future__ import annotations

import pytest

from agentic_research import __version__
from agentic_research.cli import main


def test_version_flag_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"arp {__version__}"


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: arp" in capsys.readouterr().out


def test_unknown_argument_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--no-such-flag"])
    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
