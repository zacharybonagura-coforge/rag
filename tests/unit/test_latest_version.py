from retrieve import document_family, version_key


def latest_docs(rows: list[tuple[str, str]]) -> set[tuple[str, str]]:
    best: dict[str, tuple[str, str]] = {}
    for document, version in rows:
        family = document_family(document)
        if family not in best or version_key(version) > version_key(best[family][1]):
            best[family] = (document, version)
    return set(best.values())


def test_version_key_orders_two_digit_releases() -> None:
    assert version_key("10.0") > version_key("9.0")
    assert version_key("2.10") > version_key("2.9")
    assert version_key("2.0") > version_key("1.0")
    assert version_key("") == (0,)


def test_latest_docs_keeps_highest_gru_and_other_v1() -> None:
    rows = [
        ("gru-minion-handbook-v1", "1.0"),
        ("gru-minion-handbook-v2", "2.0"),
        ("nefario-lab-safety-v1", "1.0"),
        ("girls-house-rules-v1", "1.0"),
    ]
    assert latest_docs(rows) == {
        ("gru-minion-handbook-v2", "2.0"),
        ("nefario-lab-safety-v1", "1.0"),
        ("girls-house-rules-v1", "1.0"),
    }
