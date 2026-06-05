HIGH_LABEL: str = "高い"
LOW_LABEL: str = "低い"
PARSE_FAILED_KEY: str = "parse_failed"
SKIPPED_KEY: str = "skipped"


def create_stats() -> dict[str, int]:
    # ---------------------------------------------------------
    # Initialize judgement counters.
    # ---------------------------------------------------------
    return {HIGH_LABEL: 0, LOW_LABEL: 0, PARSE_FAILED_KEY: 0, SKIPPED_KEY: 0}


def parse_judgement_label(judgement: str) -> str | None:
    # ---------------------------------------------------------
    # Parse the final judgement label from the AI response.
    # ---------------------------------------------------------
    for line in judgement.splitlines():
        if not line.startswith("判定:"):
            continue

        label = line.replace("判定:", "", 1).strip()

        if label in {HIGH_LABEL, LOW_LABEL}:
            return label

    return None


def update_stats(stats: dict[str, int], judgement: str) -> None:
    # ---------------------------------------------------------
    # Count high, low, or parse failed judgement.
    # ---------------------------------------------------------
    label = parse_judgement_label(judgement)

    if label is None:
        stats[PARSE_FAILED_KEY] += 1
        return

    stats[label] += 1
