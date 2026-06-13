from src.progress_display import format_token_count


def test_format_token_count_uses_compact_units() -> None:
    # ---------------------------------------------------------
    # Verify that token counts are shown in compact K/M/B units.
    # ---------------------------------------------------------
    assert format_token_count(999) == "999 token"
    assert format_token_count(1_500) == "1.5K token"
    assert format_token_count(2_500_000) == "2.5M token"
    assert format_token_count(3_400_000_000) == "3.4B token"
