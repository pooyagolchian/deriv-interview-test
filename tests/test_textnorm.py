from evalharness import textnorm


def test_stopwords_removed_but_negations_kept():
    toks = textnorm.tokens("The user can not view the password")
    assert "the" not in toks and "can" not in toks   # stopwords dropped
    assert "not" in toks                              # negation preserved
    assert "user" in toks and "view" in toks and "password" in toks


def test_stemming_unifies_plurals():
    assert textnorm._stem("withdrawals") == "withdrawal"
    assert textnorm._stem("reviews") == "review"
    assert textnorm._stem("statements") == "statement"
    assert textnorm._stem("policies") == "policy"
    assert textnorm._stem("screenshots") == "screenshot"
    # non-plural 'ss' words are left intact
    assert textnorm._stem("address") == "address"


def test_content_tokens_is_a_set():
    ct = textnorm.content_tokens("bank bank statement statement")
    assert ct == {"bank", "statement"}


def test_word_level_not_substring():
    # 'reset' must not be considered present just because 'preset' contains those letters
    assert "reset" not in textnorm.content_tokens("preset configuration")


def test_normalize_spaces():
    assert textnorm.normalize_spaces("  A\tScreenshot   is\nfine ") == "a screenshot is fine"


def test_empty_input():
    assert textnorm.tokens("") == []
    assert textnorm.content_tokens("   ") == set()
