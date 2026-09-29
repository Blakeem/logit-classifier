"""Unit tests for the toolkit, which need no model weights."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from logit_classifier import ChoiceAnswer, NoulAnswer, UnsupportedModelError, VisionUnsupportedError
from logit_classifier import tags as legacy_tags
from logit_classifier.backends import comfy_clip
from logit_classifier.toolkit import comfyui
from logit_classifier.toolkit import tags as toolkit_tags
from logit_classifier.toolkit.comfyui import (
    IMAGE_SLOT,
    PromptTags,
    TagSettings,
    chat,
    clip_generate,
    comfy_classifier,
    english_text,
    fit_picture,
    generate_text,
    prompt_tags,
    routed_progress_bars,
    run_outcome,
    send_status,
    skip_resident_loads,
    stop_when,
    tag_picture,
    thing_scores,
    token_fill,
)
from logit_classifier.toolkit.comfyui import classifier as comfyui_classifier
from logit_classifier.toolkit.comfyui import tagger as comfyui_tagger
from logit_classifier.toolkit.tags import (
    BROAD_THRESHOLDS,
    MAX_CANDIDATES,
    PRESENCE_STATEMENT,
    STRICT_THRESHOLDS,
    MergedCandidates,
    PresenceThresholds,
    clean_item,
    complete_tags,
    drop_subsets,
    drop_unfinished_tag,
    head_noun,
    in_prompt,
    merge_candidates,
    normalize_item,
    parse_candidates,
    passed_by_origin,
    presence_statement,
    prompt_word_forms,
    repeated_block,
    split_prompt,
    tag_stop,
)


class TestCleanItem:
    @pytest.mark.parametrize(("text", "item"), [
        ("- cat", "cat"),
        ("* dog", "dog"),
        ("\u2022 bird", "bird"),
        ("1. fish", "fish"),
        ("2) frog", "frog"),
        ("3.", ""),
        ("2.5 liter bottle", "2.5 liter bottle"),
        ("3d render", "3d render"),
        ('"red hat."', "red hat"),
        ("\u201cCat\u201d", "cat"),
        ("\u2018dog\u2019", "dog"),
        ("`fox`", "fox"),
        ("  Red   Wooden\tChair.. ", "red wooden chair"),
        ("flag of the U.S.A.", "flag of the u.s.a."),
        ('"D.C.."', "d.c."),
        ("- e.g.", "e.g."),
        ("u.s.a", "u.s.a."),
        ("u.s.a flag", "u.s.a. flag"),
        ("a.b .", "a.b."),
        ("w.b.yeats.", "w.b.yeats"),
        ("vitamin C.", "vitamin c"),
    ])
    def test_cleans_one_item(self, text, item):
        assert clean_item(text) == item


class TestParseCandidates:
    def test_splits_on_commas_semicolons_and_newlines(self):
        assert parse_candidates("cat, dog; bird\nfish") == ["cat", "dog", "bird", "fish"]

    def test_strips_list_bullets(self):
        text = "- cat\n* dog\n\u2022 bird\n1. fish\n2) frog"

        assert parse_candidates(text) == ["cat", "dog", "bird", "fish", "frog"]

    def test_keeps_a_number_that_is_not_a_bullet(self):
        assert parse_candidates("3d render, 2 cats") == ["3d render", "2 cats"]

    def test_strips_quotes_and_trailing_periods(self):
        assert parse_candidates('"cat", \'dog\', bird.') == ["cat", "dog", "bird"]
        assert parse_candidates('"red hat."') == ["red hat"]

    def test_drops_a_think_block(self):
        assert parse_candidates("<think>\nmaybe a cat, a dog\n</think>\n\nfox, tree") == ["fox", "tree"]
        assert parse_candidates("<think>still thinking, cat") == []

    def test_drops_a_leading_assistant_line(self):
        assert parse_candidates("assistant\nfox, tree") == ["fox", "tree"]
        assert parse_candidates("fox\nassistant") == ["fox", "assistant"]

    def test_lowercases_and_collapses_whitespace(self):
        assert parse_candidates("Red   Wooden\tChair") == ["red wooden chair"]

    def test_drops_empty_and_long_items(self):
        seven_words = "one two three four five six seven"
        six_words = "one two three four five six"
        long_item = "a" * 61
        edge_item = "a" * 60

        candidates = parse_candidates(f"cat,, ,{seven_words},{six_words},{long_item},{edge_item}")
        assert candidates == ["cat", six_words, edge_item]

    def test_two_spellings_of_an_initialism_are_one_tag(self):
        assert parse_candidates("U.S.A., u.s.a, flag") == ["u.s.a.", "flag"]

    def test_removes_duplicates_keeping_first_order(self):
        assert parse_candidates("dog, cat, Dog, cat.") == ["dog", "cat"]

    def test_caps_the_candidate_count(self):
        text = ", ".join(f"tag {index}" for index in range(MAX_CANDIDATES + 5))

        candidates = parse_candidates(text)
        assert len(candidates) == MAX_CANDIDATES
        assert candidates[-1] == f"tag {MAX_CANDIDATES - 1}"

    def test_takes_its_limits_as_arguments(self):
        text = "cat, black cat, big black cat, dog, bird"

        assert parse_candidates(text, max_words=2) == ["cat", "black cat", "dog", "bird"]
        assert parse_candidates(text, max_chars=3) == ["cat", "dog"]
        assert parse_candidates(text, max_candidates=2) == ["cat", "black cat"]


class TestDropUnfinishedTag:
    def test_keeps_the_text_through_the_last_separator(self):
        assert drop_unfinished_tag("cat, dog; bi") == "cat, dog"
        assert drop_unfinished_tag("cat\ndo") == "cat"

    def test_a_text_with_no_separator_is_all_unfinished(self):
        assert drop_unfinished_tag("cat") == ""


class TestCompleteTags:
    def test_skips_the_tag_still_being_written(self):
        assert complete_tags("man, man") == ["man"]
        assert complete_tags("man, man,") == ["man", "man"]

    def test_cleans_and_drops_empty_tags(self):
        assert complete_tags("- Cat,, \n\u201cDog\u201d;fox") == ["cat", "dog"]


class TestRepeatedBlock:
    @pytest.mark.parametrize(("tags", "size"), [
        (["man", "man"], 1),
        (["man", "man with hat"], 0),
        (["man", "manatee"], 0),
        (["sky", "a", "b", "c", "a", "b", "c"], 3),
        (["a", "b", "c", "d", "a", "b", "c", "d"], 4),
        (["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"], 0),
        (["a", "b", "a"], 0),
        ([], 0),
    ])
    def test_finds_only_a_back_to_back_block(self, tags, size):
        assert repeated_block(tags) == size

    def test_takes_the_largest_block_as_an_argument(self):
        tags = ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"]

        assert repeated_block(tags, max_block=5) == 5
        assert repeated_block(["a", "b", "a", "b"], max_block=1) == 0

    def test_returns_the_smallest_block(self):
        assert repeated_block(["a", "a", "a", "a"]) == 1


class TestSplitPrompt:
    def test_splits_on_every_divider(self):
        prompt = 'cat, dog; fox! owl? (hat) [cup] {pen} bee|ant/elk "yak" <emu>\nrat\rbat'

        assert split_prompt(prompt) == [
            "cat", "dog", "fox", "owl", "hat", "cup", "pen", "bee", "ant", "elk", "yak", "emu", "rat", "bat",
        ]

    def test_splits_on_a_period_or_colon_outside_a_number(self):
        assert split_prompt("A cat. A dog: a fox") == ["a cat", "a dog", "a fox"]
        assert split_prompt("a 2.5 liter bottle, 16:9 frame") == ["a 2.5 liter bottle", "16:9 frame"]
        assert split_prompt("version 2. next") == ["version 2", "next"]

    def test_keeps_an_initialism_whole(self):
        assert split_prompt("a flag of the U.S.A., a cowboy") == ["a flag of the u.s.a.", "a cowboy"]
        assert split_prompt("Washington D.C. at night. Rainy street.") == [
            "washington d.c. at night", "rainy street",
        ]
        assert split_prompt("e.g. clouds, 10 a.m. sunrise, u.s.a") == ["e.g. clouds", "10 a.m. sunrise", "u.s.a."]
        # A decomposed accent is composed first, so it cannot end the initialism mid letter.
        assert split_prompt("a.e\u0301.c") == ["a.\u00e9.c."]

    def test_splits_a_long_prompt_in_linear_time(self):
        prompt = "a.b " * 20000

        started = time.perf_counter()
        assert split_prompt(prompt) == ["a.b. " * 19999 + "a.b."]
        assert time.perf_counter() - started < 1.0

    def test_a_single_letter_or_a_word_before_a_period_still_divides(self):
        assert split_prompt("vitamin C. Blue sky") == ["vitamin c", "blue sky"]
        assert split_prompt("file.txt, a.bc") == ["file", "txt", "a", "bc"]
        assert split_prompt("Mr. Smith") == ["mr", "smith"]

    def test_strips_edge_characters_and_collapses_whitespace(self):
        assert split_prompt("  *Big   Red\tHat_ , 'cat' , `-dog-`") == ["big red hat", "cat", "dog"]

    def test_drops_a_part_with_no_letter_and_a_repeat(self):
        assert split_prompt("cat, 42, --, Cat, 3.5, dog") == ["cat", "dog"]

    def test_counts_a_non_ascii_letter(self):
        assert split_prompt("caf\u00e9, 12") == ["caf\u00e9"]

    def test_empty_prompt(self):
        assert split_prompt("") == []


class TestNormalizeItem:
    @pytest.mark.parametrize(("text", "item"), [
        ("the old man", "old man"),
        ("and the man", "man"),
        ("the", "the"),
        ("a an", "an"),
        ("with a hat of wool", "hat of wool"),
        ("- The Old Man.", "old man"),
        ("theater", "theater"),
        ("", ""),
    ])
    def test_removes_leading_filler_words(self, text, item):
        assert normalize_item(text) == item


class TestDropSubsets:
    def test_drops_an_item_another_item_holds(self):
        assert drop_subsets(["herbs", "hanging dried herbs"]) == ["hanging dried herbs"]

    @pytest.mark.parametrize("rule", ["words", "head-noun"])
    def test_keeps_the_first_of_equal_word_sets(self, rule):
        assert drop_subsets(["red car", "car red"], rule=rule) == ["red car"]

    def test_splits_words_on_hyphens(self):
        assert drop_subsets(["tomato", "sun-dried tomato", "dried sun"], rule="head-noun") == [
            "sun-dried tomato", "dried sun",
        ]

    def test_folds_a_hyphenated_superset_with_the_same_head(self):
        assert drop_subsets(["tomato", "sun-dried tomato"], rule="head-noun") == ["sun-dried tomato"]

    def test_keeps_a_whole_beside_its_named_parts(self):
        assert drop_subsets(["cat", "cat ear", "cat paw"], rule="head-noun") == ["cat", "cat ear", "cat paw"]

    def test_drops_only_into_a_superset_with_the_same_head(self):
        assert drop_subsets(["sword", "sword hilt", "steel sword"], rule="head-noun") == ["sword hilt", "steel sword"]

    def test_keeps_a_whole_beside_a_longer_part(self):
        assert drop_subsets(["table", "wooden table leg"], rule="head-noun") == ["table", "wooden table leg"]

    def test_keeps_order_and_unrelated_items(self):
        assert drop_subsets(["sky", "red car", "car", "tree", "old tree"]) == ["sky", "red car", "old tree"]

    def test_empty_list(self):
        assert drop_subsets([]) == []

    def test_the_words_rule_drops_a_whole_beside_its_part(self):
        assert drop_subsets(["cat", "cat ear"], rule="words") == ["cat ear"]
        assert drop_subsets(["table", "wooden table leg"], rule="words") == ["wooden table leg"]

    def test_the_words_rule_splits_words_on_hyphens_whatever_the_head(self):
        assert drop_subsets(["tomato", "sun-dried tomato", "dried sun"], rule="words") == ["sun-dried tomato"]

    def test_the_default_is_the_words_rule(self):
        items = ["cat", "cat ear", "tomato", "sun-dried tomato", "dried sun"]

        assert drop_subsets(items) == drop_subsets(items, rule="words") == ["cat ear", "sun-dried tomato"]

    def test_the_head_noun_rule_keeps_a_whole_beside_its_part(self):
        assert drop_subsets(["cat", "cat ear"], rule="head-noun") == ["cat", "cat ear"]

    def test_the_head_noun_rule_reads_the_head_before_a_break_word(self):
        assert drop_subsets(["cheese", "wheels of cheese"], rule="head-noun") == ["cheese", "wheels of cheese"]
        assert drop_subsets(["cheese", "wheels of cheese"], rule="words") == ["wheels of cheese"]

    @pytest.mark.parametrize("rule", ["words", "head-noun"])
    def test_both_rules_fold_a_head_into_its_modified_phrase(self, rule):
        assert drop_subsets(["signage", "signage in chinese"], rule=rule) == ["signage in chinese"]
        assert drop_subsets(["wheels", "wheels of cheese"], rule=rule) == ["wheels of cheese"]

    @pytest.mark.parametrize("rule", ["head", "Words", ""])
    def test_rejects_an_unknown_rule(self, rule):
        with pytest.raises(ValueError, match=f"rule {rule!r}. Use \"words\" or \"head-noun\""):
            drop_subsets([], rule=rule)


class TestHeadNoun:
    @pytest.mark.parametrize(("phrase", "head"), [
        ("wheels of cheese", "wheels"),
        ("signage in chinese", "signage"),
        ("sun-dried tomato", "tomato"),
        ("sign reading open", "sign"),
        ("cat ear", "ear"),
        ("cat", "cat"),
        ("Tea Cup with handle on table", "cup"),
    ])
    def test_takes_the_last_word_before_the_first_break_word(self, phrase, head):
        assert head_noun(phrase) == head

    @pytest.mark.parametrize(("phrase", "head"), [
        ("  Of Mice ", "of mice"),
        ("in", "in"),
        ("", ""),
        ("   ", ""),
        (" - ", "-"),
    ])
    def test_a_phrase_with_no_word_before_a_break_gives_its_text(self, phrase, head):
        assert head_noun(phrase) == head


class TestCompatibilityPath:
    def test_lists_the_names_0_2_1_published(self):
        assert set(legacy_tags.__all__) == {
            "MAX_CANDIDATES", "MAX_TAG_WORDS", "MAX_TAG_CHARS", "MAX_REPEAT_BLOCK", "clean_item",
            "parse_candidates", "drop_unfinished_tag", "complete_tags", "repeated_block", "split_prompt",
            "normalize_item", "drop_subsets",
        }

    def test_every_listed_name_is_the_toolkit_object(self):
        for name in legacy_tags.__all__:
            assert getattr(legacy_tags, name) is getattr(toolkit_tags, name), name


class TestInPrompt:
    def test_every_plural_cut_is_a_form(self):
        assert prompt_word_forms("Trees, berries and 2 boxes") == {
            "trees", "tree", "tre", "berries", "berrie", "berri", "berry", "and", "2", "boxes", "boxe", "box",
        }

    def test_a_plural_tag_matches_a_singular_prompt_word_and_back(self):
        assert in_prompt("trees", prompt_word_forms("a tree on a hill"))
        assert in_prompt("tree", prompt_word_forms("three trees on a hill"))
        assert in_prompt("berries", prompt_word_forms("a berry"))

    def test_every_content_word_needs_a_prompt_word(self):
        forms = prompt_word_forms("a red car beside a tree")

        assert in_prompt("red car", forms)
        assert not in_prompt("blue car", forms)

    def test_stop_words_need_no_prompt_word(self):
        assert in_prompt("the car with a tree", prompt_word_forms("car, tree"))

    def test_a_tag_of_stop_words_only_is_not_in_the_prompt(self):
        forms = prompt_word_forms("a cat on the mat with the hat")

        assert not in_prompt("on the", forms)
        assert not in_prompt("", forms)


class TestTagStop:
    def test_stops_at_max_tags(self):
        stop = tag_stop(max_tags=3, ungrounded_streak=0)

        assert not stop(["a", "b"])
        assert stop(["a", "b", "c"])

    def test_no_max_tags_never_stops_on_length(self):
        assert not tag_stop(max_tags=None, ungrounded_streak=0)([f"tag {index}" for index in range(500)])

    def test_stops_on_a_streak_of_tags_outside_the_prompt(self):
        stop = tag_stop(max_tags=None, ungrounded_streak=2, word_forms=prompt_word_forms("a cat and a dog"))

        assert not stop(["bird"])
        assert not stop(["cat", "bird", "dog", "fish"])
        assert stop(["cat", "bird", "fish"])

    def test_stops_on_a_repeated_block(self):
        stop = tag_stop(max_tags=None, ungrounded_streak=0)

        assert stop(["sky", "a", "b", "a", "b"])
        assert not stop(["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"])

    def test_takes_the_largest_block_as_an_argument(self):
        tags = ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"]

        assert tag_stop(max_tags=None, ungrounded_streak=0, max_block=5)(tags)
        assert not tag_stop(max_tags=None, ungrounded_streak=0, max_block=1)(["a", "b", "a", "b"])

    def test_a_streak_needs_word_forms(self):
        with pytest.raises(ValueError, match="ungrounded_streak=2 and no word_forms"):
            tag_stop(max_tags=None, ungrounded_streak=2)

    def test_a_streak_of_0_needs_no_word_forms(self):
        assert not tag_stop(max_tags=None, ungrounded_streak=0)(["bird", "fish", "owl"])


class TestMergeCandidates:
    CATEGORIES = frozenset({"objects", "people"})

    def test_model_tags_come_first_then_prompt_tags(self):
        merged = merge_candidates(["cat", "The Dog"], ["a tree", "cat"], cap=10, drop=self.CATEGORIES)

        assert merged == MergedCandidates(("cat", "dog", "tree"), ("both", "model", "prompt"), ())
        assert merged.candidates == ("cat", "dog", "tree")

    def test_drops_a_category_noun(self):
        merged = merge_candidates(["objects", "cat"], ["people"], cap=10, drop=self.CATEGORIES)

        assert merged.candidates == ("cat",)
        assert merged.dropped == (("objects", "category noun"), ("people", "category noun"))

    def test_drops_a_repeat_and_a_second_both(self):
        merged = merge_candidates(["cat", "cat."], ["tree", "a tree", "cat", "cat"], cap=10, drop=self.CATEGORIES)

        assert merged.candidates == ("cat", "tree")
        assert merged.origins == ("both", "prompt")
        assert merged.dropped == (("cat", "repeat"), ("tree", "repeat"), ("cat", "repeat"))

    def test_drops_a_new_name_past_the_cap(self):
        merged = merge_candidates(["cat", "dog"], ["tree", "dog"], cap=2, drop=self.CATEGORIES)

        assert merged.candidates == ("cat", "dog")
        assert merged.origins == ("model", "both")
        assert merged.dropped == (("tree", "over the cap"),)

    def test_skips_an_empty_name(self):
        merged = merge_candidates(["", " - ", "cat"], [""], cap=10, drop=self.CATEGORIES)

        assert merged == MergedCandidates(("cat",), ("model",), ())


class TestPassedByOrigin:
    THRESHOLDS: ClassVar[dict[str, float]] = {"model": 0.9, "both": 0.9, "prompt": 0.9999}

    def test_keeps_the_candidates_that_reach_their_origin_threshold(self):
        passed = passed_by_origin(["cat", "dog", "tree", "sky"], ["model", "prompt", "both", "prompt"],
                                  [0.9, 0.99, 0.95, 0.9999], thresholds=self.THRESHOLDS)

        assert passed == ("cat", "tree", "sky")

    def test_no_candidates(self):
        assert passed_by_origin([], [], [], thresholds={}) == ()

    def test_names_an_origin_with_no_threshold(self):
        with pytest.raises(ValueError, match="origin 'both', 'prompt'"):
            passed_by_origin(["cat", "dog", "sky"], ["prompt", "model", "both"], [1.0, 1.0, 1.0],
                             thresholds={"model": 0.9})


class TestPresenceStatement:
    @pytest.mark.parametrize("tag", ["cat", "apples", "two dogs", "people", "sand", "the moon", "salt and pepper shakers"])
    def test_inserts_the_tag_verbatim(self, tag):
        assert presence_statement(tag) == f"This image visibly contains {tag}"

    def test_strips_the_space_around_the_tag(self):
        assert presence_statement("  cat  ") == "This image visibly contains cat"

    def test_keeps_braces_in_the_tag_literally(self):
        assert presence_statement("{TAG} sign") == "This image visibly contains {TAG} sign"

    def test_the_default_wording_is_the_presence_wording(self):
        assert presence_statement("cat") == PRESENCE_STATEMENT.replace("{TAG}", "cat")

    def test_takes_the_wording_as_an_argument(self):
        assert presence_statement("owl", wording="Can you see {TAG}? Is it {TAG}?") == "Can you see owl? Is it owl?"

    def test_rejects_a_wording_with_no_tag(self):
        with pytest.raises(ValueError, match=r"lacks \{TAG\}"):
            presence_statement("cat", wording="Is there a cat?")

    @pytest.mark.parametrize("tag", ["", "   "])
    def test_rejects_an_empty_tag(self, tag):
        with pytest.raises(ValueError, match="at least one word"):
            presence_statement(tag)


class TestPresenceThresholds:
    def test_strict_is_at_least_broad_in_each_field(self):
        assert isinstance(BROAD_THRESHOLDS, PresenceThresholds)
        assert isinstance(STRICT_THRESHOLDS, PresenceThresholds)
        for broad, strict in zip(BROAD_THRESHOLDS, STRICT_THRESHOLDS, strict=True):
            assert strict >= broad


class InterruptProcessingException(BaseException):
    """Core's Cancel signal, which subclasses BaseException so an except Exception lets it through."""


class CoreProgressBar:
    """comfy.utils.ProgressBar, down to its construction surface."""

    def __init__(self, total, node_id=None):
        self.total = total


@pytest.fixture
def fake_comfy(monkeypatch):
    """comfy.model_management and comfy.utils, down to what the ComfyUI toolkit reads."""
    state = SimpleNamespace(loaded=[], interrupt_checks=0, interrupted=False)
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    utils = ModuleType("comfy.utils")

    def throw_exception_if_processing_interrupted():
        state.interrupt_checks += 1
        if state.interrupted:
            raise InterruptProcessingException

    management.current_loaded_models = state.loaded
    management.InterruptProcessingException = InterruptProcessingException
    management.throw_exception_if_processing_interrupted = throw_exception_if_processing_interrupted
    utils.ProgressBar = CoreProgressBar
    comfy.model_management = management
    comfy.utils = utils
    for name, module in (("comfy", comfy), ("comfy.model_management", management), ("comfy.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    return state


class TestComfyuiPackage:
    def test_importing_it_loads_no_host(self):
        # A subprocess, since this process may already hold torch and the fake comfy modules.
        probe = (
            "import sys; import logit_classifier.toolkit.comfyui; "
            "print([m for m in ('torch', 'transformers', 'comfy', 'server', 'comfy_execution') if m in sys.modules])"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "[]"

    def test_every_exported_name_exists(self):
        assert [name for name in comfyui.__all__ if not hasattr(comfyui, name)] == []
        assert comfyui.ComfyClipBackend is comfy_clip.ComfyClipBackend
        assert comfyui.PACKED_TOKEN_CEILING == comfy_clip.PACKED_TOKEN_CEILING

    def test_the_backend_module_no_longer_holds_the_vision_scope(self):
        assert not hasattr(comfy_clip, "shared_vision_encode")


class TestComfyClassifier:
    @pytest.fixture
    def built(self, monkeypatch):
        backends = []

        class FakeClipBackend:
            def __init__(self, clip, *, batch_branches=True):
                if not clip.qwen:
                    raise UnsupportedModelError("the library's own text")
                self.model_id = "Qwen/Qwen3-VL-4B-Instruct"
                self.sees_images = clip.vision
                self.batch_branches = batch_branches
                backends.append(self)

        monkeypatch.setattr(comfyui_classifier, "ComfyClipBackend", FakeClipBackend)
        return backends

    @pytest.mark.parametrize("batch_branches", [True, False])
    def test_builds_a_classifier_with_the_prior_off(self, built, batch_branches):
        classifier = comfy_classifier(SimpleNamespace(qwen=True, vision=True), batch_branches=batch_branches)

        assert classifier.backend is built[0]
        assert built[0].batch_branches is batch_branches
        assert classifier.config.use_prior_debias is False
        assert classifier.config.calibration_path is None

    def test_a_clip_that_is_not_qwen3_vl_names_the_node_and_chains_the_backend_error(self, built):
        with pytest.raises(UnsupportedModelError, match=r"^Logit Tagger: this CLIP is not a Qwen3-VL") as caught:
            comfy_classifier(SimpleNamespace(qwen=False, vision=True), node="Logit Tagger")

        assert isinstance(caught.value.__cause__, UnsupportedModelError)
        assert str(caught.value.__cause__) == "the library's own text"

    def test_a_clip_without_its_vision_tower_names_the_node(self, built):
        with pytest.raises(VisionUnsupportedError, match=r"^Logit Tagger: this CLIP was loaded without its vision"):
            comfy_classifier(SimpleNamespace(qwen=True, vision=False), node="Logit Tagger")

    def test_a_text_node_accepts_a_clip_without_its_vision_tower(self, built):
        classifier = comfy_classifier(SimpleNamespace(qwen=True, vision=False), needs_images=False)

        assert classifier.backend is built[0]

    def test_without_a_node_the_error_starts_with_the_fault(self, built):
        with pytest.raises(UnsupportedModelError, match=r"^this CLIP is not a Qwen3-VL"):
            comfy_classifier(SimpleNamespace(qwen=False, vision=True))


class LoadingClip:
    """comfy.sd.CLIP, down to load_model and the patcher state core's own load leaves behind."""

    def __init__(self):
        self.patcher = SimpleNamespace(
            load_device="cpu",
            patches_uuid="patches",
            model=SimpleNamespace(device="cpu", current_weight_patches_uuid="patches", model_loaded_weight_memory=1),
        )
        self.loads = []

    def load_model(self, tokens=None):
        self.loads.append(tokens)
        return self.patcher


class TestSkipResidentLoads:
    def test_a_resident_clip_returns_its_patcher_without_loading(self, fake_comfy):
        clip = LoadingClip()
        fake_comfy.loaded[:] = [SimpleNamespace(model=clip.patcher)]

        with skip_resident_loads(clip):
            assert clip.load_model("tokens") is clip.patcher
        assert clip.loads == []

    def test_a_clip_that_is_not_resident_loads(self, fake_comfy):
        clip = LoadingClip()
        fake_comfy.loaded[:] = [SimpleNamespace(model=SimpleNamespace())]

        with skip_resident_loads(clip):
            assert clip.load_model("tokens") is clip.patcher
        assert clip.loads == ["tokens"]

    @pytest.mark.parametrize("shadowed", [True, False], ids=["was-shadowed", "not-shadowed"])
    @pytest.mark.parametrize("raises", [False, True], ids=["normal-exit", "exception"])
    def test_the_original_load_is_restored(self, fake_comfy, shadowed, raises):
        clip = LoadingClip()
        instance_load = clip.load_model
        if shadowed:
            clip.load_model = instance_load

        with contextlib.suppress(RuntimeError), skip_resident_loads(clip):
            assert vars(clip)["load_model"] is not instance_load
            if raises:
                raise RuntimeError("the body blew up")

        if shadowed:
            assert vars(clip)["load_model"] is instance_load
        else:
            assert "load_model" not in vars(clip)


class TestSendStatus:
    @pytest.fixture
    def server(self, monkeypatch):
        sent = []
        module = ModuleType("server")

        class Instance:
            fail = False

            def send_progress_text(self, text, node_id):
                if self.fail:
                    raise ConnectionError("the socket closed")
                sent.append((text, node_id))

        module.PromptServer = type("PromptServer", (), {"instance": Instance()})
        monkeypatch.setitem(sys.modules, "server", module)
        return SimpleNamespace(module=module, sent=sent)

    def test_sends_the_text_under_the_node(self, server):
        send_status("7", "tagging")

        assert server.sent == [("tagging", "7")]

    def test_without_a_node_id_sends_nothing(self, server):
        send_status(None, "tagging")

        assert server.sent == []

    def test_without_the_server_module_does_nothing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "server", None)

        send_status("7", "tagging")

    def test_without_a_server_instance_does_nothing(self, server):
        server.module.PromptServer = type("PromptServer", (), {})

        send_status("7", "tagging")

        assert server.sent == []

    def test_swallows_a_failed_send(self, server):
        server.module.PromptServer.instance.fail = True

        send_status("7", "tagging")

        assert server.sent == []


class TestTokenFill:
    @pytest.mark.parametrize(("value", "fill"), [(0, 0.0), (13, 0.2), (65, 1.0), (90, 1.0)],
                             ids=["zero", "mid", "at-ratio", "past-ratio"])
    def test_fills_to_one_at_the_ratio_of_the_total(self, value, fill):
        assert token_fill(value, 100, 0.65) == pytest.approx(fill)

    @pytest.mark.parametrize("total", [0, -1])
    def test_an_empty_bar_reads_as_full(self, total):
        assert token_fill(5, total, 0.65) == 1.0


class TestRoutedProgressBars:
    def test_routes_every_bar_built_inside_and_yields_core_class(self, fake_comfy):
        import comfy.utils

        updates = []
        preview = object()

        with routed_progress_bars(lambda value, total, shown: updates.append((value, total, shown))) as core_bar:
            bar = comfy.utils.ProgressBar(10, node_id="7")
            bar.update_absolute(4, preview=preview)
            bar.update(2)
            bar.update_absolute(3, total=20)

        assert core_bar is CoreProgressBar
        assert comfy.utils.ProgressBar is CoreProgressBar
        assert (bar.node_id, bar.current, bar.total) == ("7", 3, 20)
        assert updates == [(4, 10, preview), (6, 10, None), (3, 20, None)]
        assert fake_comfy.interrupt_checks == 3

    def test_a_cancel_raises_before_the_update_is_routed(self, fake_comfy):
        import comfy.utils

        updates = []
        fake_comfy.interrupted = True

        with pytest.raises(InterruptProcessingException), routed_progress_bars(
            lambda value, total, shown: updates.append(value)
        ):
            comfy.utils.ProgressBar(10).update_absolute(1)

        assert updates == []
        assert comfy.utils.ProgressBar is CoreProgressBar

    def test_restores_core_class_after_an_exception(self, fake_comfy):
        import comfy.utils

        with pytest.raises(RuntimeError), routed_progress_bars(lambda value, total, shown: None):
            assert comfy.utils.ProgressBar is not CoreProgressBar
            raise RuntimeError("the body blew up")

        assert comfy.utils.ProgressBar is CoreProgressBar


class LateInterrupt(InterruptProcessingException):
    """A subclass of core's Cancel signal."""


class TestRunOutcome:
    @pytest.mark.parametrize(("exc_type", "text"), [
        (None, "4 images in 9.3 s"),
        (InterruptProcessingException, "cancelled"),
        (LateInterrupt, "cancelled"),
        (RuntimeError, "failed"),
        (KeyboardInterrupt, "failed"),
    ], ids=["done", "cancel", "cancel-subclass", "error", "base-exception"])
    def test_names_how_the_run_ended(self, fake_comfy, exc_type, text):
        assert run_outcome(exc_type, "4 images in 9.3 s") == text


STOP_ID = 0


class Token:
    """A [1, 1] token tensor, down to what core's decode loop and the stop_when sampler read."""

    def __init__(self, value):
        self.value = value
        self.shape = (1, 1)

    def reshape(self, *shape):
        return [self.value]

    def new_full(self, shape, fill):
        return Token(fill)


class GeneratingClip:
    """comfy.sd.CLIP, down to tokenize, generate and decode, with a canned reply of id_count ids."""

    def __init__(self, reply="cat, dog", id_count=3, reads_images=True):
        self.reply = reply
        self.id_count = id_count
        self.reads_images = reads_images
        self.tokenize_calls = []
        self.generate_calls = []

    def tokenize(self, text, **kwargs):
        row = [(text, 1.0)]

        self.tokenize_calls.append((text, kwargs))
        if self.reads_images:
            row = [({"type": "image", "data": image}, 1.0) for image in kwargs.get("images", [])] + row
        return {"qwen3vl_4b": [row]}

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((tokens, kwargs))
        return list(range(self.id_count))

    def decode(self, ids):
        return self.reply


class TextTokenizerClip(GeneratingClip):
    """A CLIP whose tokenizer takes no images keyword."""

    def tokenize(self, text):
        return super().tokenize(text)


class ScriptedTransformer:
    """Returns one scripted token per sample_token call, then the stop token, as core's sampler does."""

    def __init__(self, script):
        self.script = [*script, STOP_ID]
        self.model = SimpleNamespace(config=SimpleNamespace(stop_tokens=[STOP_ID]))

    def sample_token(self, logits, *args, **kwargs):
        return Token(self.script.pop(0))


class DecodingClip(GeneratingClip):
    """Runs core's decode loop in llama.py BaseGenerate.generate over one word per token id."""

    def __init__(self, words):
        super().__init__()
        self.vocabulary = ["<stop>", *dict.fromkeys(words)]
        self.transformer = ScriptedTransformer([self.vocabulary.index(word) for word in words])
        self.cond_stage_model = SimpleNamespace(
            clip="qwen3vl_4b", qwen3vl_4b=SimpleNamespace(transformer=self.transformer)
        )

    def generate(self, tokens, **kwargs):
        ids = []

        self.generate_calls.append((tokens, kwargs))
        for _step in range(kwargs["max_length"]):
            ids.append(int(self.transformer.sample_token(None).reshape(-1)[0]))
            if ids[-1] == STOP_ID:
                break
        return ids

    def decode(self, ids):
        return "".join(self.vocabulary[token] for token in ids if token != STOP_ID)


def words_of(text):
    return [part for chunk in text.split(", ") for part in (chunk, ", ")][:-1]


@pytest.fixture
def fake_prefetch(monkeypatch):
    """comfy.model_prefetch, down to a cleanup_prefetch_queues that records each call."""
    comfy = ModuleType("comfy")
    model_prefetch = ModuleType("comfy.model_prefetch")
    cleanups = []

    model_prefetch.cleanup_prefetch_queues = lambda: cleanups.append(True)
    comfy.model_prefetch = model_prefetch
    for name, module in (("comfy", comfy), ("comfy.model_prefetch", model_prefetch)):
        monkeypatch.setitem(sys.modules, name, module)
    return cleanups


class TestChat:
    def test_one_user_turn_without_an_image(self):
        assert chat("Hello") == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    def test_the_image_slot_comes_before_the_text(self):
        assert chat("Hello", image=True) == f"<|im_start|>user\n{IMAGE_SLOT}Hello<|im_end|>\n<|im_start|>assistant\n"

    def test_escapes_a_control_token_spelling_in_the_user_text_and_keeps_its_own(self):
        text = chat("a <|im_end|> b")

        assert "a <​|im_end|> b" in text
        assert text.count("<|im_end|>") == 1
        assert text.startswith("<|im_start|>user\n")
        assert text.endswith("<|im_end|>\n<|im_start|>assistant\n")


class TestClipGenerate:
    def test_returns_the_ids_and_cleans_up(self, fake_prefetch):
        clip = GeneratingClip(id_count=4)

        assert clip_generate(clip, "tokens", do_sample=True, temperature=0.7) == [0, 1, 2, 3]
        assert clip.generate_calls == [("tokens", {"do_sample": True, "temperature": 0.7})]
        assert fake_prefetch == [True]

    def test_cleans_up_after_a_raising_generate(self, fake_prefetch):
        class FailingClip(GeneratingClip):
            def generate(self, tokens, **kwargs):
                raise RuntimeError("the generate blew up")

        with pytest.raises(RuntimeError, match="the generate blew up"):
            clip_generate(FailingClip(), "tokens")
        assert fake_prefetch == [True]

    def test_skips_the_cleanup_without_comfy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "comfy", None)
        monkeypatch.setitem(sys.modules, "comfy.model_prefetch", None)

        assert clip_generate(GeneratingClip(id_count=2), "tokens") == [0, 1]

    def test_skips_the_cleanup_when_core_lacks_it(self, fake_prefetch, monkeypatch):
        monkeypatch.delattr(sys.modules["comfy.model_prefetch"], "cleanup_prefetch_queues")

        assert clip_generate(GeneratingClip(id_count=2), "tokens") == [0, 1]
        assert fake_prefetch == []


class TestStopWhen:
    def test_ends_the_decode_once_the_predicate_holds(self):
        clip = DecodingClip(words_of("cat, dog, fox, owl"))
        seen = []

        def two_tags(tags):
            seen.append(list(tags))
            return len(tags) >= 2

        with stop_when(clip, two_tags):
            ids = clip.generate(None, max_length=50)
        assert clip.decode(ids) == "cat, dog"
        assert ["cat"] in seen
        assert seen[-1] == ["cat", "dog"]

    def test_is_a_no_op_for_a_clip_without_core_sampler(self):
        ran = []

        with stop_when(GeneratingClip(), lambda tags: True):
            ran.append(True)
        assert ran == [True]

    def test_is_a_no_op_for_a_model_without_stop_tokens(self):
        clip = DecodingClip(words_of("cat, dog, fox"))
        clip.transformer.model.config.stop_tokens = []

        with stop_when(clip, lambda tags: True):
            assert "sample_token" not in vars(clip.transformer)
            ids = clip.generate(None, max_length=50)
        assert clip.decode(ids) == "cat, dog, fox"

    @pytest.mark.parametrize("shadowed", [True, False], ids=["was-shadowed", "not-shadowed"])
    @pytest.mark.parametrize("raises", [False, True], ids=["normal-exit", "exception"])
    def test_the_original_sampler_is_restored(self, shadowed, raises):
        clip = DecodingClip(words_of("cat, dog"))
        instance_sampler = clip.transformer.sample_token
        if shadowed:
            clip.transformer.sample_token = instance_sampler

        with contextlib.suppress(RuntimeError), stop_when(clip, lambda tags: False):
            assert vars(clip.transformer)["sample_token"] is not instance_sampler
            clip.generate(None, max_length=50)
            if raises:
                raise RuntimeError("the body blew up")

        if shadowed:
            assert vars(clip.transformer)["sample_token"] is instance_sampler
        else:
            assert "sample_token" not in vars(clip.transformer)


class TestGenerateText:
    def test_sends_the_chat_text_and_the_image_greedy(self):
        clip = GeneratingClip(reply="fox, snow")
        image = object()

        text = generate_text(clip, "Tag it.", image=image, max_tokens=16)
        (rendered, kwargs), = clip.tokenize_calls
        (_tokens, generate_kwargs), = clip.generate_calls
        assert text == "fox, snow"
        assert rendered == chat("Tag it.", image=True)
        assert kwargs.keys() == {"images"}
        assert kwargs["images"] == [image]
        assert generate_kwargs == {"do_sample": False, "max_length": 16}

    def test_without_an_image_tokenizes_the_text_alone(self):
        clip = GeneratingClip(reply="a red fox")

        assert generate_text(clip, "Write.", max_tokens=16) == "a red fox"
        assert clip.tokenize_calls == [(chat("Write."), {})]

    @pytest.mark.parametrize(("id_count", "whole_tags", "reply"), [
        (16, True, "fox, snow"),
        (15, True, "fox, snow, red fire hyd"),
        (16, False, "fox, snow, red fire hyd"),
    ], ids=["fills-the-budget", "below-the-budget", "whole-tags-off"])
    def test_cuts_only_a_reply_that_fills_the_budget_with_whole_tags(self, id_count, whole_tags, reply):
        clip = GeneratingClip(reply="fox, snow, red fire hyd", id_count=id_count)

        assert generate_text(clip, "Tag it.", image=object(), max_tokens=16, whole_tags=whole_tags) == reply

    def test_stops_on_should_stop_and_restores_the_sampler(self):
        clip = DecodingClip(words_of("cat, dog, fox, owl"))

        text = generate_text(clip, "Tag it.", image=object(), max_tokens=50, should_stop=lambda tags: "dog" in tags)
        assert text == "cat, dog"
        assert "sample_token" not in vars(clip.transformer)

    def test_goes_through_the_graph_cleanup(self, fake_prefetch):
        generate_text(GeneratingClip(), "Tag it.", max_tokens=16)

        assert fake_prefetch == [True]

    def test_a_tokenizer_without_an_images_keyword_raises_vision_unsupported(self):
        clip = TextTokenizerClip()

        with pytest.raises(VisionUnsupportedError, match=r"^this CLIP's tokenizer does not read images") as caught:
            generate_text(clip, "Tag it.", image=object(), max_tokens=16)
        assert isinstance(caught.value.__cause__, TypeError)
        assert clip.generate_calls == []

    def test_tokens_with_no_image_entry_raise_vision_unsupported(self):
        clip = GeneratingClip(reads_images=False)

        with pytest.raises(VisionUnsupportedError, match=r"a Qwen3-VL text encoder with a Load CLIP type that keeps"):
            generate_text(clip, "Tag it.", image=object(), max_tokens=16)
        assert clip.generate_calls == []


LANGUAGE_QUESTION = "Is this text written in English?"


def verify_nouls(scores):
    return {f"This image visibly contains {tag}": p for tag, p in scores.items()}


class ReplyingClip(GeneratingClip):
    """A GeneratingClip that gives its replies in turn, one per generate."""

    def __init__(self, *replies, events=None):
        super().__init__()
        self.replies = list(replies)
        self.events = events if events is not None else []

    def generate(self, tokens, **kwargs):
        self.events.append("generate")
        self.reply = self.replies.pop(0)
        return super().generate(tokens, **kwargs)


class TaggingClassifier:
    """Classifier, down to nouls and classify, with nouls scripted by statement and p("other") by quoted tag."""

    def __init__(self, nouls=None, p_other=None, events=None, answer=ChoiceAnswer):
        self.noul_of = nouls or {}
        self.p_other = p_other or {}
        self.events = events if events is not None else []
        self.answer = answer
        self.noul_calls = []
        self.classify_calls = []

    def nouls(self, statements, *, image=None, state=""):
        self.events.append("nouls")
        self.noul_calls.append((list(statements), image, state))
        return [self.noul_of.get(statement, 0.0) for statement in statements]

    def classify(self, request, image=None):
        answers = {}

        self.events.append("classify")
        self.classify_calls.append((request, image))
        for qid, question in request.questions.items():
            p = self.p_other.get(question.instructions.split('"')[1], 0.0)
            answers[qid] = (NoulAnswer(noul=p) if self.answer is NoulAnswer else
                            ChoiceAnswer(choice="other", confidence=p, probabilities={"thing": 1.0 - p, "other": p}))
        return SimpleNamespace(answers=answers), None


class TestTagSettings:
    def test_the_defaults_hash(self):
        assert hash(TagSettings()) == hash(TagSettings())
        assert {TagSettings(): "cached"}[TagSettings()] == "cached"

    def test_both_packs_current_values_construct(self):
        logit_tagger = TagSettings(propose_cap=40, merge_cap=80, thing_check="prompt", subsets_rule="head-noun",
                                   order="score")
        tile_refine = TagSettings(propose_cap=25, merge_cap=64, thing_check="all", subsets_rule="words",
                                  order="proposed", language_question=None, echo_fallback_instruction=None)

        assert logit_tagger != tile_refine
        assert hash(logit_tagger) != hash(tile_refine)

    def test_the_defaults_are_the_presence_wording_and_the_broad_thresholds(self):
        settings = TagSettings()

        assert settings.verify_statement == PRESENCE_STATEMENT
        assert settings.verify_threshold == BROAD_THRESHOLDS.proposed
        assert settings.prompt_only_threshold == BROAD_THRESHOLDS.declared

    def test_an_empty_verify_statement_needs_no_tag(self):
        assert TagSettings(verify_statement="").verify_statement == ""

    def test_the_thing_criteria_may_hold_more_keys(self):
        criteria = (*TagSettings().thing_criteria, ("unsure", "a word with no clear referent"))

        assert TagSettings(thing_criteria=criteria).thing_criteria == criteria

    @pytest.mark.parametrize(("field", "value", "match"), [
        pytest.param("prompt_tags_instruction", "List the things.", r"a prompt_tags_instruction without \{PROMPT\}",
                     id="prompt-tags-placeholder"),
        pytest.param("translate_instruction", "Translate it.", r"a translate_instruction without \{PROMPT\}",
                     id="translate-placeholder"),
        pytest.param("thing_question", "What is it", r"a thing_question without \{TAG\}", id="thing-placeholder"),
        pytest.param("verify_statement", "This image holds it", r"a verify_statement without \{TAG\}",
                     id="verify-placeholder"),
        pytest.param("english_threshold", 0.0, r"english_threshold=0\.0\. Use a value above 0 and at most 1",
                     id="english-threshold-zero"),
        pytest.param("thing_threshold", 1.5, r"thing_threshold=1\.5\. Use a value above 0", id="thing-threshold-high"),
        pytest.param("verify_threshold", -0.1, r"verify_threshold=-0\.1\. Use", id="verify-threshold-negative"),
        pytest.param("prompt_only_threshold", float("nan"), r"prompt_only_threshold=nan\. Use",
                     id="prompt-only-threshold-nan"),
        pytest.param("propose_max_tokens", 0, r"propose_max_tokens=0\. Use 1 or more", id="propose-budget"),
        pytest.param("propose_cap", 0, r"propose_cap=0\. Use 1 or more", id="propose-cap"),
        pytest.param("merge_cap", 0, r"merge_cap=0\. Use 1 or more", id="merge-cap"),
        pytest.param("prompt_tags_max_tokens", 0, r"prompt_tags_max_tokens=0\. Use 1 or more", id="prompt-budget"),
        pytest.param("translate_max_tokens", 0, r"translate_max_tokens=0\. Use 1 or more", id="translate-budget"),
        pytest.param("ungrounded_streak", -1, r"ungrounded_streak=-1\. Use 0 or more", id="streak"),
        pytest.param("merge_cap", 24, r"merge_cap=24 below propose_cap=25", id="merge-below-propose"),
        pytest.param("thing_check", "some", r"thing_check='some'\. Use 'all' or 'prompt'", id="thing-check"),
        pytest.param("subsets_rule", "letters", r"subsets_rule='letters'\. Use 'words' or 'head-noun'",
                     id="subsets-rule"),
        pytest.param("order", "alphabetical", r"order='alphabetical'\. Use 'score' or 'proposed'", id="order"),
        pytest.param("thing_criteria", (("thing", "an object"),), r"thing_criteria with the keys \['thing'\]",
                     id="thing-criteria"),
    ])
    def test_rejects_an_invalid_value_naming_the_field(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            TagSettings(**{field: value})


class TestThingScores:
    def test_asks_each_unknown_tag_once_in_one_text_only_request(self):
        classifier = TaggingClassifier(p_other={"sunset": 0.95, "cat": 0.1})
        settings = TagSettings()
        known = {"dog": 0.2}

        scores = thing_scores(classifier, ["cat", "dog", "sunset", "cat"], settings, known)
        (request, image), = classifier.classify_calls
        assert scores == (0.1, 0.2, 0.95, 0.1)
        assert image is None
        assert request.state == ""
        assert [question.instructions for question in request.questions.values()] == [
            'What does the image tag "cat" name', 'What does the image tag "sunset" name']
        assert all(question.criteria == dict(settings.thing_criteria) for question in request.questions.values())
        assert known == {"dog": 0.2, "cat": 0.1, "sunset": 0.95}

    def test_known_tags_make_no_request(self):
        classifier = TaggingClassifier()

        assert thing_scores(classifier, ["cat"], TagSettings(), {"cat": 0.3}) == (0.3,)
        assert classifier.classify_calls == []

    def test_fills_the_tag_in_literally(self):
        classifier = TaggingClassifier()

        thing_scores(classifier, ["{0} sign"], TagSettings(), {})
        (request, _image), = classifier.classify_calls
        assert [question.instructions for question in request.questions.values()] == [
            'What does the image tag "{0} sign" name']

    def test_a_non_choice_answer_raises(self):
        with pytest.raises(TypeError, match=r"got a noul answer, not a choice"):
            thing_scores(TaggingClassifier(answer=NoulAnswer), ["cat"], TagSettings(), {})


class TestEnglishText:
    @pytest.mark.parametrize("english", [0.97, 0.5], ids=["confident", "at-the-threshold"])
    def test_an_english_prompt_skips_the_translation(self, english):
        clip = ReplyingClip()
        classifier = TaggingClassifier(nouls={LANGUAGE_QUESTION: english})

        assert english_text(clip, classifier, "a red car", TagSettings()) == "a red car"
        assert classifier.noul_calls == [([LANGUAGE_QUESTION], None, "a red car")]
        assert clip.generate_calls == []

    def test_another_language_is_translated(self):
        clip = ReplyingClip("  a red car at night \n")
        classifier = TaggingClassifier(nouls={LANGUAGE_QUESTION: 0.02})
        settings = TagSettings()

        assert english_text(clip, classifier, "un coche {rojo}", settings) == "a red car at night"
        (text, _kwargs), = clip.tokenize_calls
        (_tokens, generate_kwargs), = clip.generate_calls
        assert text == chat(settings.translate_instruction.replace("{PROMPT}", "un coche {rojo}"))
        assert generate_kwargs == {"do_sample": False, "max_length": 512}

    def test_a_translation_that_fills_its_budget_keeps_its_tail(self):
        clip = ReplyingClip("a red car, at ni")
        classifier = TaggingClassifier(nouls={LANGUAGE_QUESTION: 0.02})

        assert english_text(clip, classifier, "un coche", TagSettings(translate_max_tokens=3)) == "a red car, at ni"

    def test_a_blank_translation_gives_the_prompt_back(self):
        classifier = TaggingClassifier(nouls={LANGUAGE_QUESTION: 0.02})

        assert english_text(ReplyingClip("  \n"), classifier, "un coche", TagSettings()) == "un coche"

    def test_no_language_question_turns_the_gate_off(self):
        clip = ReplyingClip()
        classifier = TaggingClassifier()

        assert english_text(clip, classifier, "un coche", TagSettings(language_question=None)) == "un coche"
        assert classifier.noul_calls == []
        assert clip.generate_calls == []


class TestPromptTags:
    @pytest.mark.parametrize("prompt", ["", "   \n"], ids=["empty", "blank"])
    def test_a_blank_prompt_makes_no_request(self, prompt):
        clip = ReplyingClip()
        classifier = TaggingClassifier()

        assert prompt_tags(clip, classifier, prompt, TagSettings(), {}) == PromptTags(
            text="", reply="", listed=(), p_other=(), tags=())
        assert clip.tokenize_calls == []
        assert classifier.noul_calls == []
        assert classifier.classify_calls == []

    def test_lists_the_grounded_things_the_thing_check_keeps(self):
        clip = ReplyingClip("red car, tree, sunset, blue sky")
        classifier = TaggingClassifier(nouls={LANGUAGE_QUESTION: 0.99}, p_other={"sunset": 0.95, "tree": 0.2})
        settings = TagSettings()
        known = {}

        result = prompt_tags(clip, classifier, "a red car under a tree at sunset", settings, known)
        (text, _kwargs), = clip.tokenize_calls
        (_tokens, generate_kwargs), = clip.generate_calls
        assert result == PromptTags(text="a red car under a tree at sunset", reply="red car, tree, sunset, blue sky",
                                    listed=("red car", "tree", "sunset"), p_other=(0.0, 0.2, 0.95),
                                    tags=("red car", "tree"))
        assert text == chat(settings.prompt_tags_instruction.replace("{PROMPT}", "a red car under a tree at sunset"))
        assert generate_kwargs == {"do_sample": False, "max_length": 256}
        assert known == {"red car": 0.0, "tree": 0.2, "sunset": 0.95}

    def test_lists_and_grounds_from_the_translation(self):
        clip = ReplyingClip("a red car", "red car, coche")
        classifier = TaggingClassifier(nouls={LANGUAGE_QUESTION: 0.01})
        settings = TagSettings()

        result = prompt_tags(clip, classifier, "un coche rojo", settings, {})
        assert result.text == "a red car"
        assert result.listed == ("red car",)
        assert clip.tokenize_calls[1][0] == chat(settings.prompt_tags_instruction.replace("{PROMPT}", "a red car"))

    def test_a_streak_of_ungrounded_tags_ends_the_list(self):
        clip = DecodingClip(words_of("red car, blue sky, green hills, tree"))
        classifier = TaggingClassifier(nouls={LANGUAGE_QUESTION: 0.99})

        result = prompt_tags(clip, classifier, "a red car beside a tree", TagSettings(), {})
        assert result.reply == "red car, blue sky, green hills"
        assert result.listed == ("red car",)


class TestTagPicture:
    """The fake verify scores sit between 0.9 and 0.999, so each test that keeps tags pins verify_threshold=0.9."""

    @pytest.fixture(autouse=True)
    def events(self, monkeypatch):
        """The tagger's vision scope, replaced by one that records its entry and exit, so no torch is needed."""
        events = []

        @contextlib.contextmanager
        def scope(clip):
            events.append("enter")
            yield
            events.append("exit")

        monkeypatch.setattr(comfyui_tagger, "shared_vision_encode", scope)
        return events

    def test_propose_check_and_verify_share_one_vision_scope(self, events):
        clip = ReplyingClip("cat, dog", events=events)
        classifier = TaggingClassifier(nouls=verify_nouls({"cat": 0.95, "dog": 0.97}), events=events)
        picture = object()

        trace = tag_picture(clip, classifier, picture, settings=TagSettings(verify_threshold=0.9), known={}, on_stage=events.append)
        assert events == ["enter", "propose", "generate", "classify", "verify", "nouls", "exit"]
        assert clip.tokenize_calls[0][1] == {"images": [picture]}
        assert classifier.noul_calls == [(["This image visibly contains cat", "This image visibly contains dog"],
                                          picture, "")]
        assert trace.scores == (0.95, 0.97)
        assert trace.kept == ("dog", "cat")
        assert trace.propose_ms >= 0.0
        assert trace.verify_ms >= 0.0

    def test_verify_asks_the_settings_wording(self):
        clip = ReplyingClip("cat, dog")
        classifier = TaggingClassifier(nouls={"Seen: cat": 0.95, "Seen: dog": 0.97})
        picture = object()
        settings = TagSettings(verify_statement="Seen: {TAG}", verify_threshold=0.9, thing_check="prompt")

        trace = tag_picture(clip, classifier, picture, settings=settings, known={})
        assert classifier.noul_calls == [([presence_statement(tag, wording="Seen: {TAG}") for tag in ("cat", "dog")],
                                          picture, "")]
        assert trace.kept == ("dog", "cat")

    def test_propose_stops_at_the_cap(self):
        clip = DecodingClip(words_of("cat, dog, fox, owl"))
        settings = TagSettings(propose_cap=2, merge_cap=2)

        trace = tag_picture(clip, TaggingClassifier(), object(), settings=settings, known={})
        (text, _kwargs), = clip.tokenize_calls
        (_tokens, generate_kwargs), = clip.generate_calls
        assert text == chat(settings.propose_instruction, image=True)
        assert generate_kwargs == {"do_sample": False, "max_length": 128}
        assert trace.reply == "cat, dog"
        assert trace.proposed == ("cat", "dog")

    def test_a_category_noun_only_reply_is_proposed_again_with_the_fallback(self):
        clip = ReplyingClip("objects, people, animals", "cat, dog")
        settings = TagSettings()

        trace = tag_picture(clip, TaggingClassifier(), object(), settings=settings, known={})
        assert [text for text, _kwargs in clip.tokenize_calls] == [
            chat(settings.propose_instruction, image=True), chat(settings.echo_fallback_instruction, image=True)]
        assert trace.echo_reply == "objects, people, animals"
        assert trace.reply == "cat, dog"
        assert trace.proposed == ("cat", "dog")
        assert trace.candidates == ("cat", "dog")

    def test_a_reply_holding_a_tag_is_not_proposed_again(self):
        clip = ReplyingClip("objects, cat")

        trace = tag_picture(clip, TaggingClassifier(), object(), settings=TagSettings(verify_threshold=0.9), known={})
        assert len(clip.generate_calls) == 1
        assert trace.echo_reply is None
        assert trace.candidates == ("cat",)
        assert trace.dropped == (("objects", "category noun"),)

    def test_no_fallback_wording_turns_the_fallback_off(self):
        clip = ReplyingClip("objects, people")

        trace = tag_picture(clip, TaggingClassifier(), object(),
                            settings=TagSettings(echo_fallback_instruction=None), known={})
        assert len(clip.generate_calls) == 1
        assert trace.echo_reply is None
        assert trace.candidates == ()

    def test_the_all_thing_check_drops_a_non_thing(self):
        clip = ReplyingClip("cat, sunset glow")
        classifier = TaggingClassifier(nouls=verify_nouls({"cat": 0.95, "sunset glow": 0.95, "dog": 0.99999}),
                                       p_other={"sunset glow": 0.95})
        known = {"dog": 0.1}

        trace = tag_picture(clip, classifier, object(), prompt_tags=("dog",), settings=TagSettings(thing_check="all", verify_threshold=0.9),
                            known=known)
        (request, _image), = classifier.classify_calls
        assert len(request.questions) == 2
        assert trace.candidates == ("cat", "dog")
        assert trace.origins == ("model", "prompt")
        assert trace.dropped == (("sunset glow", "not a thing"),)
        assert trace.kept == ("dog", "cat")
        assert known == {"dog": 0.1, "cat": 0.0, "sunset glow": 0.95}

    def test_the_prompt_thing_check_leaves_the_candidates_alone(self):
        clip = ReplyingClip("cat, sunset glow")
        classifier = TaggingClassifier(nouls=verify_nouls({"cat": 0.95, "sunset glow": 0.95}),
                                       p_other={"sunset glow": 0.95})

        trace = tag_picture(clip, classifier, object(), settings=TagSettings(thing_check="prompt", verify_threshold=0.9), known={})
        assert classifier.classify_calls == []
        assert trace.candidates == ("cat", "sunset glow")
        assert trace.dropped == ()
        assert trace.kept == ("cat", "sunset glow")

    def test_each_origin_meets_its_own_threshold(self):
        clip = ReplyingClip("cat, owl, bird")
        classifier = TaggingClassifier(nouls=verify_nouls(
            {"cat": 0.9, "owl": 0.9, "bird": 0.89, "dog": 0.999, "fox": 0.9999}))

        trace = tag_picture(clip, classifier, object(), prompt_tags=("owl", "dog", "fox"),
                            settings=TagSettings(thing_check="prompt", verify_threshold=0.9), known={})
        assert trace.candidates == ("cat", "owl", "bird", "dog", "fox")
        assert trace.origins == ("model", "both", "model", "prompt", "prompt")
        assert trace.verified == ("cat", "owl", "fox")

    def test_an_empty_verify_statement_keeps_every_candidate_in_candidate_order(self):
        clip = ReplyingClip("cat, dog, cat ear")
        classifier = TaggingClassifier()

        trace = tag_picture(clip, classifier, object(),
                            settings=TagSettings(verify_statement="", thing_check="prompt", order="score"), known={})
        assert classifier.noul_calls == []
        assert trace.scores == (None, None, None)
        assert trace.verified == ("cat", "dog", "cat ear")
        assert trace.kept == ("dog", "cat ear")

    @pytest.mark.parametrize(("order", "kept"), [
        ("score", ("owl", "bird", "cat", "dog")),
        ("proposed", ("cat", "owl", "bird", "dog")),
    ])
    def test_orders_the_kept_tags(self, order, kept):
        clip = ReplyingClip("cat, owl, bird, dog")
        classifier = TaggingClassifier(nouls=verify_nouls({"cat": 0.95, "owl": 0.99, "bird": 0.97, "dog": 0.95}))

        trace = tag_picture(clip, classifier, object(), settings=TagSettings(thing_check="prompt", order=order, verify_threshold=0.9),
                            known={})
        assert trace.kept == kept

    @pytest.mark.parametrize(("rule", "kept"), [("words", ("cat ear",)), ("head-noun", ("cat ear", "cat"))])
    def test_drops_subsets_by_the_settings_rule(self, rule, kept):
        clip = ReplyingClip("cat, cat ear")
        classifier = TaggingClassifier(nouls=verify_nouls({"cat": 0.95, "cat ear": 0.99}))

        trace = tag_picture(clip, classifier, object(), settings=TagSettings(thing_check="prompt", subsets_rule=rule, verify_threshold=0.9),
                            known={})
        assert trace.verified == ("cat", "cat ear")
        assert trace.kept == kept


class TestFitPicture:
    @pytest.fixture
    def torch(self):
        return pytest.importorskip("torch")

    def test_shrinks_to_the_cap_keeping_the_aspect(self, torch):
        assert tuple(fit_picture(torch.rand(1, 400, 600, 3), 60_000).shape) == (1, 200, 300, 3)

    def test_takes_the_area_mean(self, torch):
        image = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).reshape(1, 2, 2, 1).repeat(1, 1, 1, 3)

        assert torch.equal(fit_picture(image, 1), torch.full((1, 1, 1, 3), 1.5))

    @pytest.mark.parametrize("max_pixels", [20_000, 1_000_000], ids=["at-the-cap", "below-the-cap"])
    def test_leaves_a_picture_within_the_cap_unchanged(self, torch, max_pixels):
        image = torch.rand(1, 100, 200, 3)

        assert torch.equal(fit_picture(image, max_pixels), image)

    def test_keeps_the_first_three_channels(self, torch):
        image = torch.rand(1, 40, 40, 4)

        assert torch.equal(fit_picture(image, 10_000), image[..., :3])
        assert tuple(fit_picture(image, 400).shape) == (1, 20, 20, 3)

    @pytest.mark.parametrize(("height", "width", "fitted"), [(1, 5000, (1, 100)), (5000, 1, (100, 1))],
                             ids=["wide", "tall"])
    def test_an_extreme_aspect_stays_within_the_cap(self, torch, height, width, fitted):
        assert tuple(fit_picture(torch.rand(1, height, width, 3), 100).shape) == (1, *fitted, 3)

    def test_rejects_a_cap_below_1(self, torch):
        with pytest.raises(ValueError, match=r"fit_picture got max_pixels=0\. Use 1 or more"):
            fit_picture(torch.rand(1, 2, 2, 3), 0)
