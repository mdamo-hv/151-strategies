from __future__ import annotations

from strategies151 import catalog
from strategies151.strategies.registry import implemented_keys


def test_catalog_covers_every_chapter_of_the_paper():
    chapters = {e.chapter for e in catalog.CATALOG}
    assert {str(n) for n in range(2, 21)} - {"20"} <= chapters


def test_every_catalog_strategy_key_is_registered():
    known = set(implemented_keys())
    for entry in catalog.CATALOG:
        for key in entry.strategy_keys:
            assert key in known, f"{entry.section} references unknown strategy {key}"


def test_every_registered_strategy_appears_in_the_catalog():
    referenced = {k for e in catalog.CATALOG for k in e.strategy_keys}
    for key in implemented_keys():
        assert key in referenced, f"{key} is implemented but missing from the catalog"


def test_runnable_entries_carry_strategy_keys():
    for entry in catalog.runnable():
        assert entry.strategy_keys


def test_unrunnable_entries_explain_what_they_need():
    for entry in catalog.CATALOG:
        if entry.status == "not_implemented":
            assert entry.requires, f"{entry.section} does not say what data it needs"
            assert not entry.strategy_keys


def test_substituted_entries_document_the_substitution():
    for entry in catalog.CATALOG:
        if entry.status == "substituted":
            assert entry.note, f"{entry.section} substitutes an input without saying so"


def test_sections_are_unique():
    sections = [e.section for e in catalog.CATALOG]
    assert len(sections) == len(set(sections))


def test_summary_counts_add_up():
    summary = catalog.summary()
    assert int(summary["total"].sum()) == len(catalog.CATALOG)
