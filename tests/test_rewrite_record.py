from src.rewrite_record import RewriteRecord
from src.vertex_client import GemmaResponse


def test_rewrite_record_normalizes_only_rewrite() -> None:
    # ---------------------------------------------------------
    # Verify that only generated training text is normalized.
    # ---------------------------------------------------------
    record = RewriteRecord.from_generation(
        offset=1,
        item={
            "id": "ＩＤ－１",
            "url": "https://example.com/Ａ",
        },
        text="ﾃｷｽﾄ１２３",
        prompt_type="高校",
        prompt="ﾌﾟﾛﾝﾌﾟﾄＡＢＣ",
        response=GemmaResponse(text="ﾘﾗｲﾄ１２３", output_tokens=3),
    )

    assert record.source_id == "ＩＤ－１"
    assert record.url == "https://example.com/Ａ"
    assert record.source_text == "ﾃｷｽﾄ１２３"
    assert record.prompt == "ﾌﾟﾛﾝﾌﾟﾄＡＢＣ"
    assert record.rewrite == "リライト123"
    assert record.output_chars == len("リライト123")
