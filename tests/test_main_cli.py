import pytest

import main as app


def test_parse_args_requires_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # ---------------------------------------------------------
    # Verify the provider has no default value in the CLI.
    # ---------------------------------------------------------
    monkeypatch.setattr("sys.argv", ["main.py"])

    with pytest.raises(SystemExit):
        app.parse_args()


def test_parse_args_accepts_digital_ocean_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify DigitalOcean can be selected from the CLI.
    # ---------------------------------------------------------
    monkeypatch.setattr("sys.argv", ["main.py", "--provider", "digital-ocean"])

    args = app.parse_args()

    assert args.provider == "digital-ocean"
