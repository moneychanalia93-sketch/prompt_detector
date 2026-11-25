import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import heuristic_features  # noqa: E402


def test_stock_phrases_counted():
    f = heuristic_features("Please delve into a comprehensive, detailed analysis. Additionally, ensure that it is tailored.")
    assert f["stock_phrase_count"] >= 5


def test_informal_and_lowercase():
    f = heuristic_features("pls help u idk how to do this lol")
    assert f["informal_count"] == 4
    assert f["starts_lowercase"] == 1
    assert f["no_end_punct"] == 1


def test_bullets_and_colons():
    f = heuristic_features("Tasks:\n- one\n- two\n1. three")
    assert f["bullet_lines"] == 3
    assert f["colon_rate"] > 0


def test_em_dash_rate():
    f = heuristic_features("This is fine \u2014 really \u2014 truly fine.")
    assert f["em_dash_rate"] > 0
    assert heuristic_features("no dashes here.")["em_dash_rate"] == 0


def test_ttr_and_word_length():
    f = heuristic_features("the the the the")
    assert f["type_token_ratio"] == 0.25
    assert heuristic_features("abcd efgh")["avg_word_len"] == 4


def test_burstiness_uniform_vs_varied():
    uniform = heuristic_features("One two three. Four five six. Seven eight nine.")
    varied = heuristic_features("Hi. This sentence is a lot longer than the first one was. Ok.")
    assert uniform["burstiness"] == 0
    assert varied["burstiness"] > uniform["burstiness"]


def test_empty_input_is_safe():
    f = heuristic_features("")
    assert f["length_words"] == 0
    assert f["type_token_ratio"] == 0
