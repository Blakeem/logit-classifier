"""Unit tests that need no model weights."""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from logit_classifier import __version__
from logit_classifier.config import ANSWER_PREFILL
from logit_classifier.labels import (
    MAX_LABELS_PER_BRANCH,
    LabelBoundaryError,
    plan_branches,
    verify_label_ids,
)
from logit_classifier.prompt import (
    ESCAPE_LABEL,
    IMAGE_MARKER,
    build_branches,
    prefix_content,
)
from logit_classifier.schema import SchemaError, parse_questions, parse_request
from logit_classifier.scoring import (
    choice_confidence,
    combine_escape,
    expected_score,
    normalise_levels,
    restricted_softmax,
    score_confidence,
)
from logit_classifier.tags import (
    MAX_CANDIDATES,
    clean_item,
    complete_tags,
    drop_subsets,
    normalize_item,
    parse_candidates,
    repeated_block,
    split_prompt,
)
from logit_classifier.vision import ImageError, extract_image

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestSchema:
    def test_accepts_published_jev_body(self):
        request = parse_request(load("quickstart_request.json"))
        assert [q.type for q in request.questions.values()] == ["choice", "score", "noul"]

    def test_choice_criteria_keeps_null_description(self):
        request = parse_request(
            {"state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": "why"}}}}
        )
        assert request.questions["q"].criteria == {"a": None, "b": "why"}

    def test_noul_criteria_uses_reserved_words_as_keys(self):
        request = parse_request(
            {"state": "x", "questions": {"q": {"type": "noul", "criteria": {"true": "T", "false": "F"}}}}
        )
        assert request.questions["q"].criteria.true_ == "T"
        assert request.questions["q"].criteria.false_ == "F"

    def test_instructions_may_be_structured(self):
        request = parse_request(
            {"state": "x", "questions": {"q": {"type": "noul", "instructions": {"question": "same person?"}}}}
        )
        assert request.questions["q"].instructions == {"question": "same person?"}

    @pytest.mark.parametrize("body", [
        {"state": "x", "questions": {}},
        {"state": "x", "questions": {"q": {"type": "choice", "criteria": {"only": None}}}},
        {"state": "x", "questions": {"q": {"type": "score", "criteria": ["one"]}}},
        {"state": "x", "questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}},
        {"state": "x", "questions": {"q": {"type": "bool", "criteria": {}}}},
        {"questions": {"q": {"type": "noul"}}},
    ])
    def test_rejects_malformed(self, body):
        with pytest.raises(SchemaError):
            parse_request(body)

    def test_rejects_over_jev_option_cap(self):
        criteria = {f"opt{i}": None for i in range(256)}
        with pytest.raises(SchemaError):
            parse_request(
                {"state": "x", "questions": {"q": {"type": "choice", "criteria": criteria}}})


class TestBranchPlan:
    @pytest.mark.parametrize("count", [2, 3, 26, 52, 53, 89, 104, 255])
    def test_every_option_is_covered_exactly_once(self, count):
        plan = plan_branches(count)
        covered = [i for group in plan.groups for i in group]
        assert sorted(covered) == list(range(count))

    def test_no_split_below_the_label_limit(self):
        assert plan_branches(MAX_LABELS_PER_BRANCH).split is False
        assert plan_branches(MAX_LABELS_PER_BRANCH + 1).split is True
        assert plan_branches(MAX_LABELS_PER_BRANCH, abstain=True).split is True

    def test_split_groups_leave_room_for_the_escape_label(self):
        plan = plan_branches(255)
        assert max(len(g) for g in plan.groups) <= MAX_LABELS_PER_BRANCH - 1

    def test_rejects_what_two_levels_cannot_hold(self):
        with pytest.raises(LabelBoundaryError):
            plan_branches(MAX_LABELS_PER_BRANCH * MAX_LABELS_PER_BRANCH)


class TestLabelIds:
    """The trailing-space check the model gate proves on weights, held here without them."""

    class MergingEncoder:
        # Merges a space into the character after it, as the shipped BPE tokenizers do.
        def __init__(self):
            self.vocab: dict[str, int] = {}

        def encode(self, text):
            return [self.vocab.setdefault(piece, len(self.vocab))
                    for piece in re.findall(r" ?\S| ", text)]

    def test_a_trailing_space_prefill_is_rejected(self):
        with pytest.raises(LabelBoundaryError):
            verify_label_ids(self.MergingEncoder(), "Answer: ", 3)

    def test_the_shipped_prefill_gives_one_distinct_id_per_label(self):
        label_ids = verify_label_ids(self.MergingEncoder(), ANSWER_PREFILL, 3)
        assert len(set(label_ids)) == 3

    def test_the_shipped_prefill_does_not_end_in_a_space(self):
        assert ANSWER_PREFILL.rstrip() == ANSWER_PREFILL


class TestBranchExpansion:
    def test_question_types_produce_expected_branch_counts(self):
        request = parse_request(load("quickstart_request.json"))
        branches = build_branches(request.questions)
        kinds = [b.kind for b in branches]
        assert kinds == ["choice", "score_joint", "noul"]

    def test_independent_scoring_makes_one_branch_per_level(self):
        request = parse_request(load("quickstart_request.json"))
        branches = build_branches(request.questions, score_method="independent")
        assert sum(b.kind == "score_level" for b in branches) == 3

    def test_split_choice_adds_an_escape_label_to_each_group(self):
        criteria = {f"opt{i}": None for i in range(89)}
        request = parse_request(
            {"state": "x", "questions": {"q": {"type": "choice", "criteria": criteria}}}
        )
        branches = build_branches(request.questions)
        assert len(branches) == 2
        for branch in branches:
            assert branch.label_count == len(branch.targets) + 1
            assert "none of these" in branch.suffix_text

    def test_question_id_never_reaches_the_prompt(self):
        request = parse_request(
            {"state": "x", "questions": {"secret_identifier": {"type": "noul", "instructions": "urgent?"}}}
        )
        assert "secret_identifier" not in build_branches(request.questions)[0].suffix_text


class TestVision:
    def _png(self, colour):
        image = Image.new("RGB", (8, 8), colour)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def test_a_plain_state_carries_no_image(self):
        assert extract_image("just text") == ("just text", None)
        assert extract_image({"a": 1}) == ({"a": 1}, None)

    def test_a_data_url_is_lifted_out_of_the_state(self):
        state = {"image": "data:image/png;base64," + self._png("red"), "note": "hello"}
        remainder, image = extract_image(state)
        assert remainder == {"note": "hello"}
        assert image.size == (8, 8)

    def test_a_state_of_only_an_image_still_renders(self):
        remainder, image = extract_image({"screenshot": self._png("blue")})
        assert remainder == "(see image)"
        assert image is not None

    def test_unreadable_images_are_rejected_at_the_boundary(self):
        with pytest.raises(ImageError):
            extract_image({"image": "data:image/png;base64,"})
        with pytest.raises(ImageError):
            extract_image({"image": "no-such-file.png"})

    def test_a_decompression_bomb_is_an_image_error(self, monkeypatch):
        # PIL raises DecompressionBombError outside OSError, which reached the service
        # as a plain-text 500 instead of a rejected state.
        image = Image.new("RGB", (16, 16), "red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
        with pytest.raises(ImageError):
            extract_image({"image": encoded})

    def test_a_file_path_is_refused_when_paths_are_off(self, monkeypatch, tmp_path):
        path = tmp_path / "real.png"
        Image.new("RGB", (8, 8), "red").save(path)

        def forbidden(*args, **kwargs):
            raise AssertionError("the file was probed or opened")

        monkeypatch.setattr(Path, "is_file", forbidden)
        monkeypatch.setattr(Image, "open", forbidden)
        with pytest.raises(ImageError, match=r"state\.image"):
            extract_image({"image": str(path)}, allow_paths=False)

    def test_a_unc_path_never_touches_the_filesystem(self, monkeypatch):
        # Probing a UNC path makes Windows contact that host with the service's credentials.
        def forbidden(*args, **kwargs):
            raise AssertionError("the UNC path reached the filesystem")

        monkeypatch.setattr(Path, "is_file", forbidden)
        with pytest.raises(ImageError, match=r"state\.screenshot"):
            extract_image({"screenshot": r"\\host\share\x.png"}, allow_paths=False)

    def test_a_data_url_still_works_when_paths_are_off(self):
        _, image = extract_image({"image": "data:image/png;base64," + self._png("red")},
                                 allow_paths=False)
        assert image is not None

    def test_the_marker_reaches_the_prompt_only_when_an_image_does(self):
        assert IMAGE_MARKER not in prefix_content("text", False)
        assert IMAGE_MARKER in prefix_content("text", True)


class TestAbstain:
    def _question(self, options):
        return parse_request(
            {"state": "x", "questions": {"q": {"type": "choice",
                                              "criteria": dict.fromkeys(options)}}}
        ).questions

    def test_off_by_default_in_build_branches(self):
        branches = build_branches(self._question(["a", "b", "c"]))
        assert branches[0].kind == "choice"
        assert branches[0].label_count == 3

    def test_adds_one_label_to_an_unsplit_question(self):
        branches = build_branches(self._question(["a", "b", "c"]), abstain=True)
        assert len(branches) == 1
        assert branches[0].kind == "member"
        assert branches[0].label_count == 4
        assert branches[0].targets == (0, 1, 2)
        assert ESCAPE_LABEL in branches[0].suffix_text

    def test_split_questions_already_carry_it(self):
        options = [f"option {i}" for i in range(120)]
        plain = build_branches(self._question(options))
        asking = build_branches(self._question(options), abstain=True)
        assert [b.label_count for b in plain] == [b.label_count for b in asking]

    @pytest.mark.parametrize("count", [51, 52, 53])
    def test_the_escape_label_never_pushes_a_branch_over_the_limit(self, count):
        options = [f"option {i}" for i in range(count)]
        branches = build_branches(self._question(options), abstain=True)
        covered = [target for branch in branches for target in branch.targets]
        assert all(b.label_count <= MAX_LABELS_PER_BRANCH for b in branches)
        assert sorted(covered) == list(range(count))


class TestScoring:
    @pytest.mark.parametrize("probabilities,published", [
        ([0.85, 0.0, 0.15], 0.78),
        ([0.0, 1.0, 0.0], 1.0),
    ])
    def test_choice_confidence_reproduces_published_examples(self, probabilities, published):
        # The docs publish worked examples but not the formula, and this one tracks
        # them to within a rounding step.
        assert choice_confidence(np.array(probabilities)) == pytest.approx(published, abs=0.01)

    def test_choice_confidence_endpoints(self):
        assert choice_confidence(np.array([0.25] * 4)) == 0.0
        assert choice_confidence(np.array([1.0, 0.0, 0.0, 0.0])) == 1.0

    @pytest.mark.parametrize("probabilities", [
        [0.0, 0.14, 0.86, 0.0, 0.0],
        [0.5, 0.5],
        [0.2] * 5,
        [0.4, 0.6, 0.0],
    ])
    def test_score_confidence_matches_the_official_formula(self, probabilities):
        # Restated from system_one_adapter._utils.confidence_metrics so that any
        # change to our version has to be deliberate.
        mode = max(range(len(probabilities)), key=probabilities.__getitem__)
        spread = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
        centre = (len(probabilities) - 1) / 2
        uniform = sum(abs(i - centre) for i in range(len(probabilities))) / len(probabilities)
        official = max(0.0, 1.0 - spread / uniform)
        assert score_confidence(np.array(probabilities)) == pytest.approx(official)

    def test_score_confidence_separates_near_levels_from_far_ones(self):
        # Levels are ordinal, and the choice formula reads only the peak, so it
        # cannot tell these two apart.
        near = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
        far = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        assert score_confidence(near) > score_confidence(far)
        assert choice_confidence(near) == choice_confidence(far)

    def test_published_score_arithmetic(self):
        levels = np.array([0.0, 0.14, 0.86, 0.0, 0.0])
        assert round(expected_score(levels), 2) == 1.86

    def test_temperature_widens_without_reordering(self):
        z = np.array([8.0, 2.0, 1.0])
        sharp = restricted_softmax(z)
        soft = restricted_softmax(z, temperature=2.5)
        assert soft[0] < sharp[0]
        assert list(np.argsort(-soft)) == list(np.argsort(-sharp))

    def test_prior_subtraction_shifts_mass_away_from_the_favoured_label(self):
        z = np.array([1.0, 1.0])
        biased = np.log(np.array([0.8, 0.2]))
        debiased = restricted_softmax(z, log_prior=biased - biased.mean())
        assert debiased[1] > debiased[0]

    def test_levels_normalise_and_survive_all_zero(self):
        assert normalise_levels(np.array([0.2, 0.6])).sum() == pytest.approx(1.0)
        assert normalise_levels(np.array([0.0, 0.0])).tolist() == [0.5, 0.5]

    def test_escape_combination_sums_to_one(self):
        groups = [np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.1, 0.8])]
        combined, _ = combine_escape(groups)
        assert len(combined) == 4
        assert combined.sum() == pytest.approx(1.0)

    def test_escape_weighting_favours_the_group_that_did_not_decline(self):
        confident = np.array([0.6, 0.35, 0.05])
        declining = np.array([0.05, 0.05, 0.90])
        combined, _ = combine_escape([confident, declining])
        assert combined[:2].sum() > combined[2:].sum()

    def test_escape_falls_back_when_every_group_declines(self):
        both = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])]
        combined, abstain = combine_escape(both)
        assert combined.sum() == pytest.approx(1.0)
        assert np.isfinite(combined).all()
        assert abstain == pytest.approx(1.0)

    def test_abstain_needs_every_group_to_decline(self):
        # One group holding the answer means the question is answerable, however
        # hard the other group declines.
        holding = np.array([0.7, 0.2, 0.1])
        declining = np.array([0.02, 0.03, 0.95])
        assert combine_escape([holding, declining])[1] == pytest.approx(0.1)
        assert combine_escape([declining, declining])[1] == pytest.approx(0.95)

    def test_abstain_does_not_shrink_as_groups_multiply(self):
        # A product would read five honest half declines as 0.03 and never fire.
        half = np.array([0.25, 0.25, 0.5])
        assert combine_escape([half] * 5)[1] == pytest.approx(0.5)

    def test_abstain_is_the_escape_mass_for_a_single_group(self):
        assert combine_escape([np.array([0.5, 0.2, 0.3])])[1] == pytest.approx(0.3)


class TestPriorStore:
    def test_a_prior_from_another_prompt_shape_is_discarded(self, tmp_path):
        from logit_classifier.calibrate import PriorStore

        path = tmp_path / "calibration.json"
        original = PriorStore(path, fingerprint="shape-a", min_observations=1)
        for _ in range(4):
            original.observe("choice", np.array([0.9, 0.1]))
        original.save()

        same = PriorStore(path, fingerprint="shape-a", min_observations=1)
        assert same.stats() == {"choice:2": 4}

        changed = PriorStore(path, fingerprint="shape-b", min_observations=1)
        assert changed.stats() == {}

    def test_prior_stays_unused_until_enough_observations(self, tmp_path):
        from logit_classifier.calibrate import PriorStore

        store = PriorStore(tmp_path / "c.json", fingerprint="f", min_observations=3)
        store.observe("choice", np.array([0.9, 0.1]))
        assert store.log_prior("choice", 2) is None
        store.observe("choice", np.array([0.9, 0.1]))
        store.observe("choice", np.array([0.9, 0.1]))
        assert store.log_prior("choice", 2) is not None

    def test_running_mean_tracks_the_observed_distribution(self, tmp_path):
        from logit_classifier.calibrate import PriorStore

        store = PriorStore(tmp_path / "c.json", fingerprint="f", min_observations=1)
        store.observe("choice", np.array([1.0, 0.0]))
        store.observe("choice", np.array([0.0, 1.0]))
        prior = store.log_prior("choice", 2)
        assert prior == pytest.approx([0.0, 0.0], abs=1e-9)


    def test_a_store_with_no_path_learns_in_memory_and_writes_nothing(self, tmp_path):
        from logit_classifier.calibrate import PriorStore

        store = PriorStore(None, fingerprint="f", min_observations=1)
        store.observe("choice", np.array([0.9, 0.1]))
        store.save()
        assert store.stats() == {"choice:2": 1}
        assert store.log_prior("choice", 2) is not None
        assert list(tmp_path.iterdir()) == []

    def test_a_bucket_that_contradicts_its_own_key_is_discarded(self, tmp_path):
        from logit_classifier.calibrate import PriorStore

        path = tmp_path / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "fingerprint": "f",
                    "means": {"choice:2": [0.4, 0.3, 0.3], "choice:3": None, "choice:4": "x",
                              "binary:2": [0.6, 0.4]},
                    "counts": {"choice:2": 9, "choice:3": 9, "choice:4": 9, "binary:2": 9},
                }
            ),
            encoding="utf-8",
        )
        store = PriorStore(path, fingerprint="f", min_observations=1)
        assert store.stats() == {"binary:2": 9}
        store.observe("choice", np.array([0.5, 0.5]))
        assert store.stats()["choice:2"] == 1

    def test_a_non_decimal_width_is_discarded(self, tmp_path):
        from logit_classifier.calibrate import PriorStore

        path = tmp_path / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "fingerprint": "f",
                    "means": {"choice:①": [0.5, 0.5]},
                    "counts": {"choice:①": 9},
                }
            ),
            encoding="utf-8",
        )
        assert PriorStore(path, fingerprint="f", min_observations=1).stats() == {}

    def test_a_path_with_no_final_component_saves_nothing(self, tmp_path):
        from logit_classifier.calibrate import PriorStore

        store = PriorStore(Path(tmp_path.anchor), fingerprint="f", min_observations=1)
        store.observe("choice", np.array([0.9, 0.1]))
        store.save()
        assert store.stats() == {"choice:2": 1}

    def test_a_failed_rename_leaves_no_staged_file(self, tmp_path):
        from logit_classifier.calibrate import PriorStore

        path = tmp_path / "priors"
        path.mkdir()
        store = PriorStore(path, fingerprint="f", min_observations=1)
        store.observe("choice", np.array([0.9, 0.1]))
        for _ in range(3):
            store.save()
        assert list(tmp_path.iterdir()) == [path]


class TestConfigEnv:
    def test_an_unknown_score_method_is_rejected(self, monkeypatch):
        from logit_classifier.config import Config

        monkeypatch.setenv("LOGIT_SCORE_METHOD", "Joint")
        with pytest.raises(ValueError, match="LOGIT_SCORE_METHOD"):
            Config.from_env()

    def test_the_library_default_writes_no_prior(self):
        from logit_classifier.config import Config

        assert Config().calibration_path is None

    def test_the_served_model_id_follows_the_package_version(self):
        # A hand-written copy would drift from __version__ on any bump that missed it.
        from logit_classifier.config import Config

        assert Config().served_model_id == f"logit-classifier-{__version__}"

    def test_a_switch_accepts_only_0_and_1(self, monkeypatch):
        # "false" used to read as True, which silently kept batching on.
        from logit_classifier.config import Config, ConfigError

        monkeypatch.setenv("LOGIT_ABSTAIN", "false")
        with pytest.raises(ConfigError, match="LOGIT_ABSTAIN"):
            Config.from_env()

    def test_a_switch_set_to_0_turns_off(self, monkeypatch):
        from logit_classifier.config import Config

        monkeypatch.setenv("LOGIT_BATCH_BRANCHES", "0")
        assert Config.from_env().batch_branches is False

    def test_a_bad_temperature_names_its_variable(self, monkeypatch):
        from logit_classifier.config import Config, ConfigError

        monkeypatch.setenv("LOGIT_TEMPERATURE", "abc")
        with pytest.raises(ConfigError, match="LOGIT_TEMPERATURE"):
            Config.from_env()

    def test_zero_permutations_from_the_environment_is_rejected(self, monkeypatch):
        from logit_classifier.config import Config, ConfigError

        monkeypatch.setenv("LOGIT_PERMUTATIONS", "0")
        with pytest.raises(ConfigError, match="LOGIT_PERMUTATIONS"):
            Config.from_env()

    @pytest.mark.parametrize("raw", ["-1", "inf"])
    def test_bad_temperature_from_the_environment_names_the_variable(self, monkeypatch, raw):
        from logit_classifier.config import Config, ConfigError

        monkeypatch.setenv("LOGIT_TEMPERATURE", raw)
        with pytest.raises(ConfigError, match="LOGIT_TEMPERATURE"):
            Config.from_env()

    @pytest.mark.parametrize("fields", [
        {"temperature": 0},
        {"temperature": -1.0},
        {"temperature": float("nan")},
        {"score_method": "Joint"},
        {"permutations": 0},
        {"max_batch_rows": 0},
    ])
    def test_direct_construction_is_validated(self, fields):
        # A ComfyUI pack builds Config itself and never passes through from_env.
        from logit_classifier.config import Config, ConfigError

        with pytest.raises(ConfigError, match=rf"Config\.{next(iter(fields))}"):
            Config(**fields)

    def test_a_config_error_catches_as_both_bases(self):
        from logit_classifier.config import ConfigError
        from logit_classifier.errors import LogitClassifierError

        assert isinstance(ConfigError("x"), LogitClassifierError)
        assert isinstance(ConfigError("x"), ValueError)


class StubBackend:
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    label_ids: ClassVar[list[int]] = list(range(52))
    sees_images = False

    def render(self, system, user, prefill, *, open_ended=False):
        body = f"<s>{system}<u>{user}"
        return body if open_ended else f"{body}<e>{prefill}"

    def encode(self, text):
        return list(text.encode("utf-8"))

    def encode_prefix(self, text, image=None):
        return self.encode(text), {}

    def score(self, prefix_ids, suffix_ids, label_counts, vision=None):
        from logit_classifier.backends.base import BranchLogits

        return [BranchLogits(z=np.zeros(count), candidate_mass=1.0) for count in label_counts]


class TestPriorGate:
    def request(self):
        return parse_request(
            {
                "state": "the card was declined",
                "questions": {"q": {"type": "choice",
                                    "criteria": {"yes": "it failed", "no": "it went through"}}},
            }
        )

    def test_disabling_the_prior_stops_observing_and_writing(self, tmp_path):
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        path = tmp_path / "calibration.json"
        config = Config(use_prior_debias=False, calibration_path=path)
        classifier = Classifier(config, StubBackend())

        classifier.classify(self.request())
        assert classifier.priors.stats() == {}
        assert list(tmp_path.iterdir()) == []

    def test_a_backend_whose_renders_disagree_is_named(self, tmp_path):
        from logit_classifier.backends.base import BackendContractError
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        class DriftingBackend(StubBackend):
            def render(self, system, user, prefill, *, open_ended=False):
                if open_ended:
                    return f"<s>{system}<u>{user}"
                return f"<turn><s>{system}<u>{user}<e>{prefill}"

        classifier = Classifier(
            Config(calibration_path=tmp_path / "c.json"), DriftingBackend()
        )
        with pytest.raises(BackendContractError, match="DriftingBackend"):
            classifier.classify(self.request())


class TestSchemaErrors:
    def test_the_rejection_names_the_field_that_failed(self):
        with pytest.raises(SchemaError) as caught:
            parse_request({"state": "x", "questions": {"q": {"type": "score", "criteria": ["one"]}}})
        assert caught.value.field == "questions.q.criteria"

    def test_an_unknown_field_is_named_rather_than_the_body(self):
        with pytest.raises(SchemaError) as caught:
            parse_request({"state": "x", "questions": {}, "temperature": 2.0})
        assert caught.value.field == "temperature"

    def test_a_missing_state_is_reported_against_state(self):
        with pytest.raises(SchemaError) as caught:
            parse_request({"questions": {"q": {"type": "noul"}}})
        assert caught.value.field == "state"

    def test_a_question_map_validates_on_its_own(self):
        questions = parse_questions({"q": {"type": "noul", "instructions": "urgent?"}})
        assert questions["q"].type == "noul"

    def test_a_noul_needs_no_criteria(self):
        assert parse_questions({"q": {"type": "noul"}})["q"].criteria is None

    def test_a_number_is_not_prose(self):
        with pytest.raises(SchemaError):
            parse_request({"state": 7, "questions": {"q": {"type": "noul"}}})

    def test_the_default_model_is_filled_in(self):
        request = parse_request({"state": "x", "questions": {"q": {"type": "noul"}}})
        assert request.model == "logit-latest"

    def test_a_response_serialises_in_the_published_field_order(self):
        from logit_classifier.schema import ChoiceAnswer, SystemOneResponse, Usage

        answer = ChoiceAnswer(choice="a", confidence=0.5, probabilities={"a": 0.5, "b": 0.5})
        payload = SystemOneResponse(model="m", answers={"q": answer}, usage=Usage()).to_dict()
        assert list(payload) == ["model", "answers", "usage"]
        assert list(payload["answers"]["q"]) == [
            "type", "choice", "confidence", "probabilities", "abstain"
        ]


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

    def test_keeps_the_first_of_equal_word_sets(self):
        assert drop_subsets(["red car", "car red"]) == ["red car"]

    def test_splits_words_on_hyphens(self):
        assert drop_subsets(["tomato", "sun-dried tomato", "dried sun"]) == ["sun-dried tomato"]

    def test_keeps_order_and_unrelated_items(self):
        assert drop_subsets(["sky", "red car", "car", "tree", "old tree"]) == ["sky", "red car", "old tree"]

    def test_empty_list(self):
        assert drop_subsets([]) == []


class TestCoreIsolation:
    def test_importing_the_package_loads_no_host(self):
        # Prime directive 4. A ComfyUI pack imports this package beside ComfyUI's own
        # pinned CUDA build, so a stray top-level torch import would be a hard break.
        # A subprocess is the only honest check, since pytest has already imported PIL.
        probe = (
            "import sys; import logit_classifier; import logit_classifier.tags; "
            "print([m for m in ('torch', 'transformers', 'fastapi', 'PIL') if m in sys.modules])"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                check=True)
        assert result.stdout.strip() == "[]"


class TestDeterminismWindow:
    """Prime directive 3 needs torch globals pinned, and a ComfyUI host needs them back."""

    HF_SOURCE: ClassVar[Path] = (
        Path(__file__).resolve().parents[1] / "src" / "logit_classifier" / "backends" / "hf.py"
    )

    @staticmethod
    def _snapshot(torch):
        # mkldnn is here because set_float32_matmul_precision writes it too. Listing
        # only what the window restores would make this test a mirror of the code.
        # The getters are reached by name for the same reason hf.py guards them, since
        # a ComfyUI venv supplies its own torch and this package never pins one.
        sdp_allowed = getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", None)
        cuda_fp32 = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
        # The coarse getter raises once a host has set a slot directly.
        matmul = torch.get_float32_matmul_precision() if cuda_fp32 is None else None
        return {
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
            "bf16": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "matmul": matmul,
            "cuda_fp32": cuda_fp32,
            "mkldnn_fp32": getattr(torch.backends.mkldnn.matmul, "fp32_precision", None),
            "sdp_reduction": sdp_allowed() if sdp_allowed else None,
            "fp16_acc": getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None),
        }

    @staticmethod
    def _apply(torch, state):
        torch.backends.cudnn.benchmark = state["benchmark"]
        torch.backends.cudnn.deterministic = state["deterministic"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = state["bf16"]
        if state["matmul"] is not None:
            torch.set_float32_matmul_precision(state["matmul"])
        if state["cuda_fp32"] is not None:
            torch.backends.cuda.matmul.fp32_precision = state["cuda_fp32"]
            torch.backends.mkldnn.matmul.fp32_precision = state["mkldnn_fp32"]
        if state["sdp_reduction"] is not None:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(state["sdp_reduction"])
        if state["fp16_acc"] is not None:
            torch.backends.cuda.matmul.allow_fp16_accumulation = state["fp16_acc"]

    @pytest.fixture
    def torch_host(self):
        # hf.py imports torch and transformers, which the hf extra supplies and CI omits.
        pytest.importorskip("transformers")
        torch = pytest.importorskip("torch")
        saved = self._snapshot(torch)
        yield torch
        self._apply(torch, saved)

    # A pristine host is what ComfyUI leaves, and it is the baseline where restoring
    # through set_float32_matmul_precision alone leaves the mkldnn slot on "ieee".
    # A host writing one slot directly is the case where the coarse getter raises.
    @pytest.mark.parametrize("baseline", ["pristine", "high", "new_api"])
    def test_the_window_pins_then_puts_every_slot_back(self, torch_host, baseline):
        from logit_classifier.backends._torch_window import _determinism

        # A host holding the opposite of every value we need. ComfyUI sets the sdp
        # reduction on at import. A bare --fast turns fp16 accumulation on as well.
        # ComfyUI-AudioSR sets float32 matmul precision to high when it is imported.
        torch_host.backends.cudnn.benchmark = True
        torch_host.backends.cudnn.deterministic = False
        torch_host.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch_host.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        torch_host.backends.cuda.matmul.allow_fp16_accumulation = True
        if baseline == "high":
            torch_host.set_float32_matmul_precision("high")
        elif baseline == "new_api":
            torch_host.backends.cuda.matmul.fp32_precision = "tf32"
        else:
            torch_host.backends.cuda.matmul.fp32_precision = "none"
            torch_host.backends.mkldnn.matmul.fp32_precision = "none"
        host = self._snapshot(torch_host)

        with _determinism():
            pinned = self._snapshot(torch_host)

        assert pinned["benchmark"] is False
        assert pinned["deterministic"] is True
        assert pinned["bf16"] is False
        if pinned["cuda_fp32"] is None:
            assert pinned["matmul"] == "highest"
        assert pinned["cuda_fp32"] == "ieee"
        assert pinned["sdp_reduction"] is False
        assert pinned["fp16_acc"] is False
        assert self._snapshot(torch_host) == host

    def test_the_window_puts_them_back_after_a_failed_pass(self, torch_host):
        from logit_classifier.backends._torch_window import _determinism

        torch_host.backends.cuda.matmul.fp32_precision = "none"
        torch_host.backends.mkldnn.matmul.fp32_precision = "none"
        torch_host.backends.cudnn.deterministic = False
        host = self._snapshot(torch_host)

        with pytest.raises(RuntimeError), _determinism():
            raise RuntimeError("the forward pass blew up")

        assert self._snapshot(torch_host) == host

    def test_the_window_puts_them_back_when_a_pin_raises(self, torch_host, monkeypatch):
        # An unusual torch build can refuse one setter after the earlier ones have landed.
        from logit_classifier.backends._torch_window import _determinism

        original = torch_host.backends.cuda.allow_fp16_bf16_reduction_math_sdp

        def refuse_pinning(enabled):
            if not enabled:
                raise RuntimeError("this build refuses the sdp pin")
            original(enabled)

        torch_host.backends.cudnn.benchmark = True
        torch_host.backends.cudnn.deterministic = False
        torch_host.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        host = self._snapshot(torch_host)
        monkeypatch.setattr(torch_host.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", refuse_pinning)

        with pytest.raises(RuntimeError, match="refuses"), _determinism():
            pass

        assert self._snapshot(torch_host) == host

    def test_the_forward_window_pins_rather_than_only_nesting(self, torch_host):
        # test_every_forward_pass_sits_inside_the_window proves lexical nesting only,
        # so a _pinned_forward that stopped calling _determinism would pass it.
        from logit_classifier.backends.hf import HFBackend

        class NoAttentionBackends:
            attention_backends = ()

        torch_host.backends.cudnn.deterministic = False
        with HFBackend._pinned_forward(NoAttentionBackends()):
            assert torch_host.backends.cudnn.deterministic is True
        assert torch_host.backends.cudnn.deterministic is False

    def test_the_fallback_path_decides_its_own_attention_backend(self, torch_host):
        # With no probed backend the host's enable flags would otherwise pick the
        # kernel, so one request could answer differently on two hosts.
        from logit_classifier.backends.hf import HFBackend

        class NoAttentionBackends:
            attention_backends = ()

        torch_host.backends.cuda.enable_flash_sdp(False)
        torch_host.backends.cuda.enable_math_sdp(False)
        try:
            with HFBackend._pinned_forward(NoAttentionBackends()):
                assert torch_host.backends.cuda.flash_sdp_enabled() is True
                assert torch_host.backends.cuda.math_sdp_enabled() is True
                assert torch_host.backends.cuda.cudnn_sdp_enabled() is True
                assert torch_host.backends.cuda.mem_efficient_sdp_enabled() is True
            assert torch_host.backends.cuda.flash_sdp_enabled() is False
            assert torch_host.backends.cuda.math_sdp_enabled() is False
        finally:
            torch_host.backends.cuda.enable_flash_sdp(True)
            torch_host.backends.cuda.enable_math_sdp(True)

    def test_every_forward_pass_sits_inside_the_window(self):
        # Scoping the globals only holds prime directive 3 if no forward pass escapes
        # the window, and no test that skips the weights can catch one that does.
        # This reads lexical nesting of a call made on self.model, so it sees
        # self.model(), .forward() and .generate() but not a call through a local
        # alias. test_the_forward_window_pins_rather_than_only_nesting covers the
        # other half, that the window still sets something.
        tree = ast.parse(self.HF_SOURCE.read_text(encoding="utf-8"))

        def model_calls(node):
            found = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and any(
                    isinstance(part, ast.Attribute)
                    and part.attr == "model"
                    and isinstance(part.value, ast.Name)
                    and part.value.id == "self"
                    for part in ast.walk(inner.func)
                ):
                    found.add(id(inner))
            return found

        def opens_the_window(node):
            return any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "_pinned_forward"
                for item in node.items
            )

        covered = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and opens_the_window(node):
                covered |= model_calls(node)

        every = model_calls(tree)
        assert every, "found no self.model call, so this guard is no longer watching anything"
        assert every == covered, f"{len(every - covered)} forward pass(es) sit outside the window"


class TestPriorBuckets:
    """The prior corrects letter bias only, so fixed-meaning positions stay out of it."""

    def _classifier(self):
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        classifier = Classifier(Config(), backend=StubBackend())
        classifier.priors.min_observations = 1
        return classifier

    def _branch(self, kind, label_count):
        from logit_classifier.prompt import Branch

        return Branch("q", kind, label_count, tuple(range(label_count)), "")

    def _logits(self, z):
        from logit_classifier.backends.base import BranchLogits

        return BranchLogits(z=np.array(z, dtype=np.float64), candidate_mass=1.0)

    def test_a_score_never_shares_a_bucket_with_a_choice_of_the_same_width(self):
        classifier = self._classifier()
        classifier._calibrated(self._branch("score_joint", 5), self._logits([4.0, 0, 0, 0, 0]))
        classifier._calibrated(self._branch("choice", 5), self._logits([0, 0, 0, 0, 4.0]))
        assert classifier.priors.stats() == {"score:5": 1, "choice:5": 1}

    def test_the_escape_label_is_neither_learned_nor_corrected(self):
        classifier = self._classifier()
        member = self._branch("member", 5)
        # Every row declines hard, which a prior over the escape would learn and undo.
        for _ in range(3):
            classifier._calibrated(member, self._logits([1.0, 0, 0, 0, 6.0]))
        calibrated, applied = classifier._calibrated(member, self._logits([1.0, 0, 0, 0, 6.0]))
        uncorrected = restricted_softmax(np.array([1.0, 0, 0, 0, 6.0]), None,
                                         classifier.temperature)
        assert applied
        assert classifier.priors.stats() == {"choice:4": 4}
        # A prior over the escape would pull this row's decline from 0.96 to 0.56.
        assert calibrated[-1] == pytest.approx(uncorrected[-1], abs=0.01)


class TestBackendIdentity:
    """The temperature follows the model the backend loaded, not a Config default."""

    def _backend(self, model_id):
        backend = StubBackend()
        backend.model_id = model_id
        return backend

    @pytest.mark.parametrize(("model_id", "expected"), [
        ("Qwen/Qwen3-VL-4B-Instruct", 1.25),
        ("Qwen/Qwen3-4B-Instruct-2507", 6.0),
        ("some/unfitted-model", 2.5),
    ])
    def test_temperature_comes_from_the_backend(self, model_id, expected):
        # A pack over a resident text encoder never sets Config.model_id, so the
        # temperature came from Config and crossed the two models, 0.088 to 0.565.
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        assert Classifier(Config(), backend=self._backend(model_id)).temperature == expected

    def test_an_explicit_temperature_still_wins(self):
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        classifier = Classifier(Config(temperature=3.0),
                                backend=self._backend("Qwen/Qwen3-4B-Instruct-2507"))
        assert classifier.temperature == 3.0

    def test_two_models_do_not_share_one_prior_bucket(self):
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        vision = Classifier(Config(), backend=self._backend("Qwen/Qwen3-VL-4B-Instruct"))
        text = Classifier(Config(), backend=self._backend("Qwen/Qwen3-4B-Instruct-2507"))
        assert vision._fingerprint() != text._fingerprint()

    @pytest.mark.parametrize("model_id", [
        "E:/models/Qwen3-VL-4B-Instruct",
        r"E:\models\Qwen3-VL-4B-Instruct",
        "E:/models/Qwen3-VL-4B-Instruct/",
    ])
    def test_a_local_directory_maps_to_its_repo(self, model_id):
        from logit_classifier.config import canonical_model_id

        assert canonical_model_id(model_id) == "Qwen/Qwen3-VL-4B-Instruct"

    def test_an_unknown_directory_is_its_own_identity(self):
        from logit_classifier.config import canonical_model_id

        assert canonical_model_id("E:/models/mystery") == "E:/models/mystery"

    def test_a_declared_canonical_id_picks_the_fitted_temperature(self):
        # A local path missed the table and got 2.5 instead of 1.25.
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        backend = self._backend("E:/weights/vl")
        backend.canonical_model_id = "Qwen/Qwen3-VL-4B-Instruct"
        assert Classifier(Config(), backend=backend).temperature == 1.25

    def test_a_backend_without_model_id_fails_the_port_check(self):
        # The point of a runtime_checkable port is that a pack can assert its own
        # object before it reaches the classifier and gets the wrong temperature.
        from logit_classifier.backends.base import Backend

        older = StubBackend()
        del type(older).model_id
        try:
            assert not isinstance(older, Backend)
        finally:
            type(older).model_id = "Qwen/Qwen3-VL-4B-Instruct"
        assert isinstance(StubBackend(), Backend)


class TestLibraryErrorBase:
    def test_every_library_error_catches_as_one_name(self):
        # A ComfyUI node turning any failure into one red message should not have to
        # name six classes, and service.py used to catch only two of them.
        from logit_classifier.backends.base import BackendContractError, VisionUnsupportedError
        from logit_classifier.deps import MissingDependencyError
        from logit_classifier.errors import LogitClassifierError
        from logit_classifier.labels import LabelBoundaryError

        every = [SchemaError("x"), ImageError("x"), LabelBoundaryError("x"),
                 MissingDependencyError("x"), BackendContractError("x"),
                 VisionUnsupportedError("x")]
        assert all(isinstance(error, LogitClassifierError) for error in every)

    def test_the_old_stdlib_bases_still_catch(self):
        from logit_classifier.deps import MissingDependencyError
        from logit_classifier.labels import LabelBoundaryError

        assert isinstance(SchemaError("x"), ValueError)
        assert isinstance(LabelBoundaryError("x"), RuntimeError)
        assert isinstance(MissingDependencyError("x"), ImportError)

    def test_a_broken_backend_contract_is_not_a_caller_error(self):
        # A host catching ValueError for a bad request must not also swallow a broken backend.
        from logit_classifier.backends.base import BackendContractError

        assert isinstance(BackendContractError("x"), RuntimeError)
        assert not isinstance(BackendContractError("x"), ValueError)


class TestModuleEntryPoint:
    """`python -m logit_classifier` is the way in when the scripts directory is off PATH."""

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        # A subprocess is the only check that proves __main__.py is importable and wired,
        # since importing it in-process never runs the __name__ guard.
        return subprocess.run([sys.executable, "-m", "logit_classifier", *args],
                              capture_output=True, text=True, check=True)

    def test_the_module_runs_and_reports_its_version(self):
        assert self._run("--version").stdout.strip() == __version__

    def test_the_module_prints_the_configuration(self):
        assert "model_id=" in self._run("config").stdout

    def test_usage_names_the_command_the_user_typed(self):
        # argparse would otherwise call it __main__.py, which no one can run.
        assert self._run("--help").stdout.startswith("usage: python -m logit_classifier")


class TestServeDependencies:
    def test_serve_names_the_extra_when_fastapi_is_missing(self, monkeypatch):
        # A None entry is how the import system spells "absent" without touching disk.
        from logit_classifier import cli
        from logit_classifier.deps import MissingDependencyError

        monkeypatch.setitem(sys.modules, "fastapi", None)
        with pytest.raises(MissingDependencyError, match=r"logit-classifier\[service\]"):
            cli.main(["serve"])


class TestModelLoading:
    """load_model is the standalone half of the Backend port the ComfyUI path uses."""

    HF_SOURCE: ClassVar[Path] = (
        Path(__file__).resolve().parent.parent / "src" / "logit_classifier" / "backends" / "hf.py"
    )

    def _capture_config(self, monkeypatch):
        """Stand in for HFBackend so the config can be read without loading weights."""
        seen: dict = {}

        class FakeBackend:
            def __init__(self, config):
                seen["config"] = config

        monkeypatch.setitem(sys.modules, "logit_classifier.backends.hf",
                            SimpleNamespace(HFBackend=FakeBackend))
        return seen

    def test_the_model_id_argument_overrides_the_config(self, monkeypatch):
        from logit_classifier import Config, load_model

        seen = self._capture_config(monkeypatch)
        load_model("some/other-model", Config(models_dir=Path("models")))
        assert seen["config"].model_id == "some/other-model"
        assert seen["config"].models_dir == Path("models")

    def test_no_arguments_loads_the_configured_default(self, monkeypatch):
        from logit_classifier import Config, load_model

        seen = self._capture_config(monkeypatch)
        load_model()
        assert seen["config"].model_id == Config().model_id

    def test_the_caller_config_is_not_mutated(self, monkeypatch):
        # Config is frozen, so an override has to replace rather than write through.
        from logit_classifier import Config, load_model

        self._capture_config(monkeypatch)
        config = Config()
        load_model("some/other-model", config)
        assert config.model_id == Config().model_id

    def test_every_loader_call_honours_the_models_directory(self):
        # A from_pretrained without cache_dir silently downloads to the global cache,
        # which no test that skips the weights would notice.
        tree = ast.parse(self.HF_SOURCE.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        ]
        assert calls, "no from_pretrained call found, the check would pass vacuously"
        for call in calls:
            keywords = {keyword.arg for keyword in call.keywords}
            assert "cache_dir" in keywords, f"from_pretrained at line {call.lineno} has no cache_dir"


class TestModelsDirEnv:
    def test_the_env_names_the_download_location(self, monkeypatch):
        from logit_classifier.config import Config

        monkeypatch.setenv("LOGIT_MODELS_DIR", "models")
        assert Config.from_env().models_dir == Path("models")

    def test_the_library_default_downloads_where_hugging_face_says(self):
        # None keeps an install from writing weights into whatever directory the
        # caller started in.
        from logit_classifier.config import Config

        assert Config().models_dir is None


class TestPublicSurface:
    """The package root is what README documents, so it has to be enough on its own."""

    def test_a_backend_author_needs_no_unpromised_import(self):
        # Before these were exported a pack had to reach into two unpromised modules,
        # or hardcode "Answer: (" and break every label when it drifted.
        import logit_classifier as pkg

        for name in ("Backend", "BranchLogits", "BackendContractError", "ANSWER_PREFILL",
                     "MAX_LABELS_PER_BRANCH", "verify_label_ids", "verify_backend",
                     "LogitClassifierError", "COMFY_SOCKET_TYPE", "UnsupportedModelError"):
            assert name in pkg.__all__, f"{name} is not exported from the package root"
            assert hasattr(pkg, name)

    def test_prompt_internals_are_not_frozen_into_the_root(self):
        import logit_classifier as pkg

        for name in ("Branch", "build_branches", "PROMPT_VERSION"):
            assert name not in pkg.__all__, f"{name} should stay on logit_classifier.prompt"

    def test_everything_exported_actually_exists(self):
        import logit_classifier as pkg

        missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
        assert missing == []

    def test_verify_backend_catches_a_render_that_does_not_line_up(self):
        from logit_classifier import BackendContractError, verify_backend

        class DropsThePrefix(StubBackend):
            def render(self, system, user, prefill, *, open_ended=False):
                return f"<u>{user}" if open_ended else f"<s>{system}<u>{user}<e>{prefill}"

        with pytest.raises(BackendContractError):
            verify_backend(DropsThePrefix())

    def test_verify_backend_renders_the_closed_probe_over_a_suffix(self):
        # Probing both renders over the same body passes a backend the real path
        # rejects, because _suffix_ids renders closed over the state plus a branch.
        from logit_classifier import BackendContractError, verify_backend

        class LengthTagged(StubBackend):
            def render(self, system, user, prefill, *, open_ended=False):
                body = f"<s>{system}<u len={len(user)}>{user}"
                return body if open_ended else f"{body}<e>{prefill}"

        with pytest.raises(BackendContractError):
            verify_backend(LengthTagged())

    def test_verify_backend_names_a_missing_model_id(self):
        # Without this the failure is a bare AttributeError from Classifier, which
        # names neither the port nor the calibration consequence.
        from logit_classifier import BackendContractError, verify_backend

        class NoIdentity(StubBackend):
            model_id = None

        with pytest.raises(BackendContractError, match="model_id"):
            verify_backend(NoIdentity())


class TestBranchPacking:
    """Chunking by suffix length, which decides padding waste and must stay pure."""

    def _packer(self, batch_branches=True, max_batch_rows=32):
        # Borrow the two methods rather than load a model, since neither touches one.
        pytest.importorskip("transformers")
        pytest.importorskip("torch")
        from logit_classifier.backends.hf import HFBackend

        class Packer:
            _pack_chunks = HFBackend._pack_chunks
            _rows_per_chunk = HFBackend._rows_per_chunk
            _kv_bytes_per_token = 147456
            config = SimpleNamespace(batch_branches=batch_branches,
                                     max_batch_rows=max_batch_rows)

        return Packer()

    def _suffixes(self, lengths):
        return [list(range(n)) for n in lengths]

    # The caller's order, not a sorted one. A wide choice question first, then
    # nouls and scores interleaved, which is what makes the sort do any work.
    MIXED: ClassVar[list[int]] = [782, 33, 33, 78, 34, 156, 33, 33, 78, 357,
                                  34, 33, 78, 81, 33, 33, 78, 33, 33, 33]

    def test_every_branch_lands_in_exactly_one_chunk(self):
        suffix_ids = self._suffixes(self.MIXED)
        groups = self._packer()._pack_chunks(suffix_ids, 916)
        assert sorted(i for group in groups for i in group) == list(range(len(suffix_ids)))

    def test_the_request_packs_into_exactly_this_grouping(self):
        # Prime directive 3. Pinning the grouping rather than comparing two calls,
        # since a pure function called twice in one process always agrees with itself.
        suffix_ids = self._suffixes(self.MIXED)
        groups = self._packer()._pack_chunks(suffix_ids, 916)
        assert [[len(suffix_ids[i]) for i in group] for group in groups] == [
            [33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 78, 78, 78, 78, 81],
            [156, 357],
            [782],
        ]

    def test_equal_widths_never_trip_the_token_ceiling(self):
        # The ceiling stops a wide branch dragging narrow ones out to its width.
        # A request of one shape has no padding to pay for, so it must chunk on the
        # row cap alone. 32 branches of 78 tokens is 2496, which is over the ceiling.
        for count, width in ((32, 65), (32, 78), (20, 156), (10, 357), (10, 1600)):
            suffix_ids = self._suffixes([width] * count)
            packer = self._packer()
            rows = packer._rows_per_chunk(90, width)
            expected = [list(range(start, min(start + rows, count)))
                        for start in range(0, count, rows)]
            assert packer._pack_chunks(suffix_ids, 90) == expected, f"{count} x {width} re-chunked"

    def test_chunks_run_in_order_of_suffix_length(self):
        # This is the property the sort exists for, and the only one here that
        # fails if the sort is removed while the ceiling stays.
        suffix_ids = self._suffixes(self.MIXED)
        groups = self._packer()._pack_chunks(suffix_ids, 916)
        lengths = [len(suffix_ids[i]) for group in groups for i in group]
        assert lengths == sorted(lengths)

    def test_packing_cuts_padded_tokens_against_request_order(self):
        suffix_ids = self._suffixes(self.MIXED)
        packer = self._packer()
        packed = self._tokens(packer._pack_chunks(suffix_ids, 916), suffix_ids)

        rows = packer._rows_per_chunk(916, max(self.MIXED))
        in_order = [list(range(start, min(start + rows, len(suffix_ids))))
                    for start in range(0, len(suffix_ids), rows)]
        assert self._tokens(in_order, suffix_ids) > packed * 4

    @staticmethod
    def _tokens(groups, suffix_ids):
        return sum(len(g) * max(len(suffix_ids[i]) for i in g) for g in groups)

    def test_a_uniform_request_is_chunked_the_way_it_already_was(self):
        # The ceiling is set above the widest chunk a uniform request uses, so a
        # request that never needed packing keeps its logits bitwise unchanged.
        for count, width in ((256, 29), (2, 641), (24, 29)):
            suffix_ids = self._suffixes([width] * count)
            groups = self._packer()._pack_chunks(suffix_ids, 90)
            assert groups == [list(range(start, min(start + 32, count)))
                              for start in range(0, count, 32)], f"{count} x {width} re-chunked"

    def test_one_branch_wider_than_the_ceiling_still_gets_a_chunk(self):
        packer = self._packer()
        from logit_classifier.backends.hf import CHUNK_TOKEN_CEILING

        suffix_ids = self._suffixes([CHUNK_TOKEN_CEILING * 2])
        assert packer._pack_chunks(suffix_ids, 90) == [[0]]

    def test_batching_off_gives_one_branch_per_chunk_in_request_order(self):
        suffix_ids = self._suffixes(self.MIXED)
        groups = self._packer(batch_branches=False)._pack_chunks(suffix_ids, 916)
        assert groups == [[i] for i in range(len(suffix_ids))]


class TestScoreScatter:
    """Packing reorders the rows, and the port promises the caller's order back."""

    def _backend(self):
        pytest.importorskip("transformers")
        torch = pytest.importorskip("torch")
        from logit_classifier.backends.base import BranchLogits
        from logit_classifier.backends.hf import HFBackend

        class Model:
            device = torch.device("cpu")

            def __call__(self, **kwargs):
                return None

        class Stub:
            score = HFBackend.score
            _pack_chunks = HFBackend._pack_chunks
            _rows_per_chunk = HFBackend._rows_per_chunk
            _pinned_forward = HFBackend._pinned_forward
            _kv_bytes_per_token = 147456
            attention_backends = ()
            config = SimpleNamespace(batch_branches=True, max_batch_rows=32)
            model = Model()

            def _score_chunk(self, prefix_ids, seed, suffix_ids, label_counts, rope_delta=None):
                # Each row carries its own suffix length, so a misplaced row shows up.
                return [BranchLogits(z=np.array([float(len(s))]), candidate_mass=1.0)
                        for s in suffix_ids]

        return Stub()

    def test_rows_come_back_in_the_order_the_caller_gave(self):
        lengths = [782, 33, 33, 78, 34, 156, 33, 33, 78, 357, 34, 33, 78, 81, 33]
        suffix_ids = [list(range(n)) for n in lengths]
        rows = self._backend().score(list(range(90)), suffix_ids, [2] * len(suffix_ids))
        assert [float(r.z[0]) for r in rows] == [float(n) for n in lengths]

    def test_one_row_per_suffix(self):
        suffix_ids = [list(range(n)) for n in (33, 782, 33)]
        rows = self._backend().score(list(range(90)), suffix_ids, [2, 2, 2])
        assert len(rows) == len(suffix_ids)

    def test_no_suffixes_returns_nothing(self):
        assert self._backend().score(list(range(90)), [], []) == []


# The Qwen chat tokens a ComfyUI Qwen tokenizer keeps whole, at their real ids.
_QWEN_SPECIAL: dict[str, int] = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|image_pad|>": 151655,
}
_FAKE_VOCAB = 151700
# The fake vision tower expands the one image entry into this many embeddings, and the
# text after it resumes this far past the image's start, as MRoPE's grid positions do.
_FAKE_PATCHES = 6
_FAKE_GRID = 3
_COMFY_CLIP_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "logit_classifier" / "backends" / "comfy_clip.py"
)


class FakeHFTokenizer:
    """Qwen2Tokenizer reduced to one id per chat token and one per character."""

    def __init__(self, chat_tokens=True):
        self.special = _QWEN_SPECIAL if chat_tokens else {}

    def encode(self, text, add_special_tokens=True):
        ids = []
        for piece in re.split(r"(<\|[a-z_]+\|>)", text):
            ids.extend([self.special[piece]] if piece in self.special else [ord(c) for c in piece])
        return ids


class FakeSDTokenizer:
    """comfy/sd1_clip.py SDTokenizer, down to the three rewrites that make it unfit for encode."""

    def __init__(self, chat_tokens=True):
        self.tokenizer = FakeHFTokenizer(chat_tokens)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        text = text.replace("\\(", "(").replace("\\)", ")")
        words = [word for word in text.split(" ") if not word.startswith("embedding:")]
        ids = self.tokenizer.encode(" ".join(words)) or [151643]
        return [[(token, 1.0) for token in ids]]


class FakeSD1Tokenizer:
    def __init__(self, name, chat_tokens=True):
        self.clip_name = name
        self.clip = name
        setattr(self, name, FakeSDTokenizer(chat_tokens))

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        inner = getattr(self, self.clip)
        return {self.clip_name: inner.tokenize_with_weights(text, return_word_ids, **kwargs)}


class FakeQwenTransformer:
    """Qwen3VL from comfy/text_encoders/qwen3vl.py and llama.py, over fixed rows.

    The packed forward's first hidden channel counts the positions each row sees, which
    is the length the generate path's row carries for the same branch without an image.
    The second channel is the token id, so a test can find each token in the sequence.
    """

    def __init__(self, visual=True):
        if visual:
            self.visual = object()
        self.fail = False
        self.forward_error = None
        self.pinned = []
        self.inference = []
        self.generated = []
        self.forwards = []
        self.logit_inputs = []
        self.image_inputs = 0
        self.deepstack = [object()]
        self.model = SimpleNamespace(forward=self.packed_forward)

    def get_input_embeddings(self):
        return self.embed

    def embed(self, ids, out_dtype=None):
        return ids.to(out_dtype).unsqueeze(-1)

    def build_image_inputs(self, embeds, embeds_info):
        import torch

        self.image_inputs += 1
        if not embeds_info:
            return None, None, None
        (image,) = embeds_info
        seq = embeds.shape[1]
        start = image["index"]
        end = start + image["size"]
        positions = torch.zeros((3, seq))
        positions[:, :start] = torch.arange(start)
        positions[:, start:end] = start
        positions[:, end:] = torch.arange(seq - end) + start + _FAKE_GRID
        visual_mask = torch.zeros((1, seq), dtype=torch.bool)
        visual_mask[0, start:end] = True
        return positions, visual_mask, self.deepstack

    def packed_forward(self, x, embeds=None, attention_mask=None, position_ids=None, deepstack_embeds=None,
                       visual_pos_masks=None, embeds_info=None):
        import torch

        self.forwards.append({
            "embeds": embeds, "attention_mask": attention_mask, "position_ids": position_ids,
            "deepstack_embeds": deepstack_embeds, "visual_pos_masks": visual_pos_masks,
            "embeds_info": embeds_info, "pinned": torch.backends.cudnn.deterministic,
            "inference": torch.is_inference_mode_enabled(),
        })
        if self.forward_error is not None:
            raise self.forward_error
        visible = attention_mask[0].sum(dim=-1).to(embeds.dtype)
        return torch.stack([visible, embeds[0, :, 0]], dim=-1).unsqueeze(0), None

    def logits(self, x):
        import torch

        self.logit_inputs.append(x)
        rows = torch.full((*x.shape[:-1], _FAKE_VOCAB), -20.0)
        rows[..., ord("A")] = x[..., 0].float()
        rows[..., ord("B")] = 0.0
        return rows

    def row(self, tokens):
        import torch

        # The winning logit is the sequence length, so every branch reads back its own row.
        row = torch.full((_FAKE_VOCAB,), -20.0)
        row[ord("A")] = float(len(tokens))
        row[ord("B")] = 0.0
        return row

    def sample_token(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, token_history,
                     generator, do_sample=True, presence_penalty=0.0, penalty_mask=None):
        import torch

        return torch.argmax(logits, dim=-1, keepdim=True)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty,
                 seed, presence_penalty=0.0):
        import torch

        self.pinned.append(torch.backends.cudnn.deterministic)
        self.inference.append(torch.is_inference_mode_enabled())
        if self.fail:
            raise RuntimeError("the forward pass blew up")
        generator = None
        penalty_mask = None
        decode_tokens = torch.empty((1, 1), dtype=torch.long)
        generated = []
        for _ in range(max_length):
            logits = self.row(tokens[0])[None, :]
            next_token = self.sample_token(logits, temperature, top_k, top_p, min_p, repetition_penalty, [],
                                           generator, do_sample=do_sample, presence_penalty=presence_penalty,
                                           penalty_mask=penalty_mask)
            decode_tokens.copy_(next_token)
            generated.append(decode_tokens[0].item())
        self.generated.append(generated)
        return generated


class FakeQwenClipModel:
    """Qwen3VLClipModel.generate: drops the weights and hands the ids to the transformer."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.processed = []
        self.vision_runs = 0

    def process_tokens(self, tokens, device):
        """SDClipModel.process_tokens, which reads bare ids and dicts and expands each image."""
        import torch

        self.processed.append(tokens)
        values = []
        info = []
        for token in tokens[0]:
            if isinstance(token, dict):
                assert set(token) == {"type", "data", "original_type"}
                assert token["type"] == "image" and token["original_type"] == "image"
                self.vision_runs += 1
                info.append({"type": "image", "index": len(values), "size": _FAKE_PATCHES, "extra": {}})
                values.extend([-1.0] * _FAKE_PATCHES)
            else:
                # Core calls .get on anything that is not an int, so a weighted pair raises there.
                assert type(token) is int, f"process_tokens reads bare ids, got {token!r}"
                values.append(float(token))
        embeds = torch.tensor(values, device=device).view(1, -1, 1)
        return embeds, torch.ones((1, len(values)), dtype=torch.long), [len(values)], info

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty,
                 seed, presence_penalty=0.0, mtp=True):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        return self.transformer.generate(tokens_only, do_sample, max_length, temperature, top_k, top_p, min_p,
                                         repetition_penalty, seed, presence_penalty=presence_penalty)


class FakeTEModel:
    def __init__(self, name, clip_model):
        self.clip_name = name
        self.clip = name
        self.events = []
        setattr(self, name, clip_model)

    def reset_clip_options(self):
        self.events.append("reset")

    def set_clip_options(self, options):
        self.events.append(options)

    def generate(self, tokens, **kwargs):
        return getattr(self, self.clip).generate(tokens, **kwargs)


class FakeClip:
    """comfy.sd.CLIP, which defines generate and decode whatever encoder it holds."""

    def __init__(self, name="qwen3vl_4b", transformer=None, chat_tokens=True):
        self.transformer = transformer if transformer is not None else FakeQwenTransformer()
        self.tokenizer = FakeSD1Tokenizer(name, chat_tokens)
        self.cond_stage_model = FakeTEModel(name, FakeQwenClipModel(self.transformer))
        self.patcher = SimpleNamespace(load_device="cpu")
        self.calls = []

    def load_model(self, tokens=None):
        self.cond_stage_model.events.append(("load", tokens))
        return self.patcher

    def tokenize(self, text, return_word_ids=False, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

    def generate(self, tokens, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.95,
                 min_p=0.0, repetition_penalty=1.0, seed=None, presence_penalty=0.0, mtp=True):
        self.calls.append({"tokens": tokens, "do_sample": do_sample, "max_length": max_length})
        return self.cond_stage_model.generate(
            tokens, do_sample=do_sample, max_length=max_length, temperature=temperature, top_k=top_k,
            top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed,
            presence_penalty=presence_penalty, mtp=mtp,
        )

    def decode(self, token_ids, skip_special_tokens=True):
        return ""


@pytest.fixture
def fake_comfy(monkeypatch):
    """comfy.model_management and comfy.ops, down to the three calls the packed pass makes."""
    state = SimpleNamespace(bf16=False, devices=[], quantized=[])
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    ops = ModuleType("comfy.ops")

    @contextlib.contextmanager
    def cuda_device_context(device):
        state.devices.append(device)
        yield

    @contextlib.contextmanager
    def use_quantized_matmul(model, device):
        state.quantized.append((model, device))
        yield

    management.should_use_bf16 = lambda device=None: state.bf16
    management.cuda_device_context = cuda_device_context
    ops.use_quantized_matmul = use_quantized_matmul
    comfy.model_management = management
    comfy.ops = ops
    for name, module in (("comfy", comfy), ("comfy.model_management", management), ("comfy.ops", ops)):
        monkeypatch.setitem(sys.modules, name, module)
    return state


class TestComfyClipBackend:
    """A Backend over a ComfyUI Qwen3-VL CLIP, proven against core's call shapes."""

    def _backend(self, clip=None, **kwargs):
        from logit_classifier.backends.comfy_clip import ComfyClipBackend

        return ComfyClipBackend(clip if clip is not None else FakeClip(), **kwargs)

    def test_importing_it_loads_no_host(self):
        # The module sits beside ComfyUI's own torch, and CI has neither torch nor comfy.
        probe = (
            "import sys; from logit_classifier.backends.comfy_clip import ComfyClipBackend; "
            "print([m for m in ('torch', 'transformers', 'comfy') if m in sys.modules])"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "[]"

    def test_a_clip_l_shaped_clip_is_rejected_at_construction(self):
        from logit_classifier.backends.base import UnsupportedModelError

        clip_l = FakeClip(name="clip_l", transformer=SimpleNamespace())
        with pytest.raises(UnsupportedModelError, match="Qwen3-VL"):
            self._backend(clip_l)

    def test_a_generating_model_without_the_qwen_chat_tokens_is_rejected(self):
        # The render is the Qwen template, which another family would read as plain text.
        from logit_classifier.backends.base import UnsupportedModelError

        with pytest.raises(UnsupportedModelError, match="Krea 2"):
            self._backend(FakeClip(name="llama3", chat_tokens=False))

    @pytest.mark.parametrize("encoder", ["qwen25_7b", "qwen35_4b", "qwen3_4b"])
    def test_another_generating_qwen_encoder_is_rejected(self, encoder):
        # Qwen2.5-VL and Qwen3.5 pass every other check, then read an image at 1D positions.
        from logit_classifier.backends.base import UnsupportedModelError

        with pytest.raises(UnsupportedModelError, match="Qwen3-VL"):
            self._backend(FakeClip(name=encoder))

    def test_it_meets_the_port_and_proves_its_labels(self):
        from logit_classifier.backends.base import Backend
        from logit_classifier.labels import LABEL_ALPHABET

        backend = self._backend()
        assert isinstance(backend, Backend)
        assert backend.label_ids == [ord(label) for label in LABEL_ALPHABET]

    @pytest.mark.parametrize(
        ("encoder", "model_id", "temperature"),
        [
            ("qwen3vl_4b", "Qwen/Qwen3-VL-4B-Instruct", 1.25),
            ("qwen3vl_8b", "Qwen/Qwen3-VL-8B-Instruct", 2.5),
            ("qwen3vl_32b", "qwen3vl_32b", 2.5),
        ],
    )
    def test_the_encoder_name_picks_the_fitted_temperature(self, encoder, model_id, temperature):
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        backend = self._backend(FakeClip(name=encoder))
        assert backend.model_id == model_id
        assert backend.canonical_model_id == model_id
        assert Classifier(Config(), backend).temperature == temperature

    def test_a_keyword_overrides_the_model_id(self):
        assert self._backend(FakeClip(name="qwen3vl_8b"), model_id="mine").model_id == "mine"

    def test_sees_images_follows_the_vision_tower(self):
        assert self._backend().sees_images is True
        assert self._backend(FakeClip(transformer=FakeQwenTransformer(visual=False))).sees_images is False

    def test_the_render_is_the_qwen_chat_template(self):
        backend = self._backend()
        opened = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU"
        assert backend.render("S", "U", "Answer: (") == (
            f"{opened}<|im_end|>\n<|im_start|>assistant\nAnswer: ("
        )
        assert backend.render("S", "U", "Answer: (", open_ended=True) == opened

    def test_the_render_matches_hf_backend(self):
        pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        from logit_classifier.backends.hf import HFBackend

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-VL-4B-Instruct", cache_dir=Path(__file__).resolve().parents[1] / "models",
                local_files_only=True,
            )
        except OSError:
            pytest.skip("the Qwen3-VL-4B-Instruct tokenizer is not in the local models cache")
        hf = SimpleNamespace(tokenizer=tokenizer)
        backend = self._backend()
        user = prefix_content("a state", True) + "Question: which?"
        for open_ended in (False, True):
            ours = backend.render("a system", user, ANSWER_PREFILL, open_ended=open_ended)
            assert ours == HFBackend.render(hf, "a system", user, ANSWER_PREFILL, open_ended=open_ended)

    def test_encode_reads_the_tokenizer_rather_than_clip_tokenize(self):
        clip = FakeClip()
        text = "a \\(b\\) embedding:foo"
        rewritten = [token for token, _ in clip.tokenize(text)["qwen3vl_4b"][0]]
        expected = FakeHFTokenizer().encode(text, add_special_tokens=False)
        assert rewritten != expected
        assert self._backend(clip).encode(text) == expected

    def test_a_prefix_without_an_image_is_plain_ids(self):
        backend = self._backend()
        assert backend.encode_prefix("text") == (backend.encode("text"), {})

    def test_an_image_on_a_text_only_clip_is_refused(self):
        from logit_classifier.backends.base import VisionUnsupportedError

        backend = self._backend(FakeClip(transformer=FakeQwenTransformer(visual=False)))
        with pytest.raises(VisionUnsupportedError):
            backend.encode_prefix(IMAGE_MARKER, SimpleNamespace(shape=(1, 8, 8, 3)))

    @pytest.mark.parametrize(("text", "count"), [("no marker", 0), (IMAGE_MARKER * 2, 2)])
    def test_an_image_needs_exactly_one_pad_token(self, text, count):
        torch = pytest.importorskip("torch")
        with pytest.raises(ImageError, match=f"holds {count} "):
            self._backend().encode_prefix(text, torch.zeros(1, 8, 8, 3))

    def test_an_image_batch_is_refused(self):
        torch = pytest.importorskip("torch")
        with pytest.raises(ImageError, match="got shape"):
            self._backend().encode_prefix(IMAGE_MARKER, torch.zeros(2, 8, 8, 3))

    def test_an_image_that_is_not_a_float_tensor_is_refused(self):
        # Core's vision path reads images.device, so a numpy array would fail deep inside it.
        torch = pytest.importorskip("torch")
        backend = self._backend()
        for image in (np.zeros((1, 8, 8, 3), dtype=np.float32), torch.zeros(1, 8, 8, 3, dtype=torch.uint8)):
            with pytest.raises(ImageError, match="float torch tensor"):
                backend.encode_prefix(IMAGE_MARKER, image)

    def test_score_puts_the_image_back_in_core_form(self):
        torch = pytest.importorskip("torch")
        clip = FakeClip()
        backend = self._backend(clip)
        image = torch.zeros(1, 8, 8, 3)
        prefix_ids, vision = backend.encode_prefix(f"Context:\n{IMAGE_MARKER}\n", image)
        backend._score_by_generate(prefix_ids, [backend.encode("x")], [2], vision)

        call = clip.calls[0]
        entry, weight = call["tokens"]["qwen3vl_4b"][0][vision["pad_index"]]
        assert entry["type"] == "image"
        assert entry["original_type"] == "image"
        assert entry["data"] is image
        assert weight == 1.0
        assert call["do_sample"] is False
        assert call["max_length"] == 1

    @pytest.mark.usefixtures("fake_comfy")
    def test_score_returns_one_row_per_branch_in_order(self):
        pytest.importorskip("torch")
        backend = self._backend()
        prefix = backend.encode("prefix")
        suffixes = [backend.encode("s" * n) for n in (5, 1, 9)]
        rows = backend.score(prefix, suffixes, [2, 3, 52])

        assert [row.z.shape[0] for row in rows] == [2, 3, 52]
        for row, suffix in zip(rows, suffixes, strict=True):
            winner = float(len(prefix) + len(suffix))
            labels = np.array([winner, 0.0] + [-20.0] * (row.z.shape[0] - 2))
            vocab = np.full(_FAKE_VOCAB, -20.0)
            vocab[ord("A")] = winner
            vocab[ord("B")] = 0.0
            expected = np.exp(np.logaddexp.reduce(labels) - np.logaddexp.reduce(vocab))
            assert row.z.dtype == np.float64
            assert np.array_equal(row.z, labels)
            assert row.candidate_mass == pytest.approx(expected, rel=1e-5)

    def test_the_sampled_token_still_reaches_decode(self):
        pytest.importorskip("torch")
        clip = FakeClip()
        backend = self._backend(clip)
        backend._score_by_generate(backend.encode("p"), [backend.encode("q")], [2])
        assert clip.transformer.generated == [[ord("A")]]

    def test_sample_token_is_restored_after_a_pass_and_after_a_failure(self):
        pytest.importorskip("torch")
        clip = FakeClip()
        backend = self._backend(clip)
        backend._score_by_generate(backend.encode("p"), [backend.encode("q")], [2])
        assert "sample_token" not in vars(clip.transformer)

        clip.transformer.fail = True
        with pytest.raises(RuntimeError, match="blew up"):
            backend._score_by_generate(backend.encode("p"), [backend.encode("q")], [2])
        assert "sample_token" not in vars(clip.transformer)

    def test_every_generate_is_pinned_and_under_inference_mode(self):
        torch = pytest.importorskip("torch")
        clip = FakeClip()
        backend = self._backend(clip)
        saved = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = False
        try:
            backend._score_by_generate(backend.encode("p"), [backend.encode("q"), backend.encode("r")], [2, 2])
            assert clip.transformer.pinned == [True, True]
            assert clip.transformer.inference == [True, True]
            assert torch.backends.cudnn.deterministic is False

            clip.transformer.fail = True
            with pytest.raises(RuntimeError):
                backend._score_by_generate(backend.encode("p"), [backend.encode("q")], [2])
            assert clip.transformer.pinned[-1] is True
            assert torch.backends.cudnn.deterministic is False
        finally:
            torch.backends.cudnn.deterministic = saved

    def test_every_generate_sits_inside_the_window(self):
        # The runtime test above sees the fake's calls only, so this reads the source.
        tree = ast.parse(_COMFY_CLIP_SOURCE.read_text(encoding="utf-8"))

        def generate_calls(node):
            return {id(inner) for inner in ast.walk(node)
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "generate"}

        covered = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "_determinism"
                for item in node.items
            ):
                covered |= generate_calls(node)

        every = generate_calls(tree)
        assert every, "found no generate call, so this guard is no longer watching anything"
        assert every == covered


@pytest.mark.usefixtures("fake_comfy")
class TestComfyPackedScore:
    """score's default path, one forward per pass of packed suffixes behind one prefix."""

    def _setup(self, *, batch_branches=True):
        torch = pytest.importorskip("torch")
        from logit_classifier.backends.comfy_clip import ComfyClipBackend

        clip = FakeClip()
        return torch, clip, ComfyClipBackend(clip, batch_branches=batch_branches)

    def _image_prefix(self, torch, backend):
        # "p", the three marker tokens and "q", so the prefix is 4 + _FAKE_PATCHES embeddings long.
        return backend.encode_prefix(f"p{IMAGE_MARKER}q", torch.zeros(1, 8, 8, 3))

    def test_the_prefix_reaches_process_tokens_as_bare_ids_and_one_image_dict(self, fake_comfy):
        torch, clip, backend = self._setup()
        image = torch.zeros(1, 8, 8, 3)
        prefix_ids, vision = backend.encode_prefix(f"Context:\n{IMAGE_MARKER}\n", image)
        backend.score(prefix_ids, [backend.encode("x")], [2], vision)

        ((sequence,),) = clip.cond_stage_model.qwen3vl_4b.processed
        pad = vision["pad_index"]
        assert sequence[pad]["data"] is image
        assert [token for i, token in enumerate(sequence) if i != pad] == [
            token for i, token in enumerate(prefix_ids) if i != pad
        ]
        assert clip.cond_stage_model.events == [
            "reset",
            ("load", {"qwen3vl_4b": [sequence]}),
            {"layer": None, "execution_device": "cpu"},
        ]
        assert fake_comfy.devices == ["cpu"]
        assert fake_comfy.quantized == [(clip.cond_stage_model, "cpu")]
        assert clip.calls == []

    def test_every_suffix_sits_after_the_image_expanded_prefix(self):
        torch, clip, backend = self._setup()
        prefix_ids, vision = self._image_prefix(torch, backend)
        backend.score(prefix_ids, [backend.encode("ab"), backend.encode("cde")], [2, 2], vision)

        (forward,) = clip.transformer.forwards
        prefix_length = len(prefix_ids) - 1 + _FAKE_PATCHES
        assert forward["embeds"].shape[1] == prefix_length + 5
        assert forward["embeds"][0, prefix_length:, 0].tolist() == [float(ord(c)) for c in "abcde"]
        pad = vision["pad_index"]
        expected_mask = [pad <= i < pad + _FAKE_PATCHES for i in range(prefix_length + 5)]
        assert forward["visual_pos_masks"][0].tolist() == expected_mask
        assert forward["deepstack_embeds"] is clip.transformer.deepstack
        assert forward["embeds_info"][0]["size"] == _FAKE_PATCHES

    def test_each_suffix_restarts_after_the_largest_image_position(self):
        torch, clip, backend = self._setup()
        prefix_ids, vision = self._image_prefix(torch, backend)
        backend.score(prefix_ids, [backend.encode("ab"), backend.encode("cde")], [2, 2], vision)

        positions = clip.transformer.forwards[0]["position_ids"]
        # The patches sit at 2, the text after them resumes at 2 + _FAKE_GRID, and the
        # prefix's largest position is 6, so each suffix restarts at 7.
        expected = [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 8, 7, 8, 9]
        assert positions.shape == (3, len(expected))
        assert [row.tolist() for row in positions] == [expected] * 3

    def test_each_suffix_restarts_after_a_text_only_prefix(self):
        _, clip, backend = self._setup()
        backend.score(backend.encode("pq"), [backend.encode("ab"), backend.encode("cde")], [2, 2])

        forward = clip.transformer.forwards[0]
        assert forward["position_ids"].tolist() == [[0, 1, 2, 3, 2, 3, 4]]
        assert forward["visual_pos_masks"] is None
        assert forward["deepstack_embeds"] is None

    def test_the_mask_lets_a_suffix_see_the_prefix_and_itself_only(self):
        _, clip, backend = self._setup()
        backend.score(backend.encode("pq"), [backend.encode("ab"), backend.encode("cde")], [2, 2])

        mask = clip.transformer.forwards[0]["attention_mask"]
        assert mask.shape == (1, 7, 7)
        assert mask[0].tolist() == [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 1],
        ]

    def test_the_row_is_read_at_each_suffix_last_position(self):
        _, clip, backend = self._setup()
        rows = backend.score(backend.encode("pq"), [backend.encode("ab"), backend.encode("cde")], [2, 3])

        (gathered,) = clip.transformer.logit_inputs
        assert gathered.shape == (2, 1, 2)
        assert gathered[:, 0, 1].tolist() == [float(ord("b")), float(ord("e"))]
        assert [row.z.tolist() for row in rows] == [[4.0, 0.0], [5.0, 0.0, -20.0]]

    @pytest.mark.parametrize("bf16", [True, False])
    def test_embeds_run_in_the_dtype_core_generate_picks(self, fake_comfy, bf16):
        torch, clip, backend = self._setup()
        fake_comfy.bf16 = bf16
        backend.score(backend.encode("pq"), [backend.encode("ab")], [2])
        expected = torch.bfloat16 if bf16 else torch.float32
        assert clip.transformer.forwards[0]["embeds"].dtype == expected

    def test_passes_split_at_the_ceiling_and_rows_return_in_request_order(self, monkeypatch):
        from logit_classifier.backends import comfy_clip

        torch, clip, backend = self._setup()
        monkeypatch.setattr(comfy_clip, "PACKED_TOKEN_CEILING", 22)
        prefix_ids, vision = self._image_prefix(torch, backend)
        lengths = [5, 7, 1, 9, 12, 20]
        rows = backend.score(prefix_ids, [[ord("s")] * n for n in lengths], [2] * len(lengths), vision)

        # The prefix is 10 embeddings, so 10+5+7 fills a pass, 10+1+9 cannot take 12,
        # and 20 cannot share a pass, so it runs alone over the ceiling.
        widths = [forward["attention_mask"].shape[-1] for forward in clip.transformer.forwards]
        assert widths == [22, 20, 22, 30]
        assert [float(row.z[0]) for row in rows] == [10.0 + n for n in lengths]

    def test_the_vision_tower_runs_once_for_a_request_of_two_passes(self, monkeypatch):
        from logit_classifier.backends import comfy_clip

        torch, clip, backend = self._setup()
        monkeypatch.setattr(comfy_clip, "PACKED_TOKEN_CEILING", 20)
        prefix_ids, vision = self._image_prefix(torch, backend)
        backend.score(prefix_ids, [[ord("s")] * 10, [ord("s")] * 10], [2, 2], vision)

        assert len(clip.transformer.forwards) == 2
        assert len(clip.cond_stage_model.qwen3vl_4b.processed) == 1
        assert clip.cond_stage_model.qwen3vl_4b.vision_runs == 1
        assert clip.transformer.image_inputs == 1

    def test_batching_off_gives_every_suffix_its_own_pass(self):
        _, clip, backend = self._setup(batch_branches=False)
        suffixes = [backend.encode(text) for text in ("ab", "cde", "f")]
        rows = backend.score(backend.encode("pq"), suffixes, [2, 2, 2])

        assert [forward["attention_mask"].shape[-1] for forward in clip.transformer.forwards] == [4, 5, 3]
        assert [float(row.z[0]) for row in rows] == [4.0, 5.0, 3.0]

    def test_a_moved_core_internal_falls_back_once_and_stays_there(self):
        _, clip, backend = self._setup()
        clip.transformer.model = SimpleNamespace()
        suffixes = [backend.encode("ab"), backend.encode("cde")]
        with pytest.warns(RuntimeWarning, match="per-branch generate path"):
            rows = backend.score(backend.encode("pq"), suffixes, [2, 2])

        assert len(clip.calls) == 2
        assert [float(row.z[0]) for row in rows] == [4.0, 5.0]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            backend.score(backend.encode("pq"), suffixes, [2, 2])
        assert len(clip.cond_stage_model.qwen3vl_4b.processed) == 1
        assert len(clip.calls) == 4

    def test_a_fallback_that_fails_too_reaches_the_caller_and_keeps_packing_on(self):
        _, clip, backend = self._setup()
        model = clip.transformer.model
        generate = clip.generate
        clip.transformer.model = SimpleNamespace()

        def broken_generate(*args, **kwargs):
            raise AttributeError("'numpy.ndarray' object has no attribute 'device'")

        clip.generate = broken_generate
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(AttributeError, match="device"):
                backend.score(backend.encode("pq"), [backend.encode("ab")], [2])
        assert backend._packed_failed is False

        clip.transformer.model = model
        clip.generate = generate
        backend.score(backend.encode("pq"), [backend.encode("ab")], [2])
        assert len(clip.transformer.forwards) == 1
        assert clip.calls == []

    def test_any_other_failure_reaches_the_caller_without_a_fallback(self):
        _, clip, backend = self._setup()
        clip.transformer.forward_error = RuntimeError("CUDA out of memory")
        with pytest.raises(RuntimeError, match="out of memory"):
            backend.score(backend.encode("pq"), [backend.encode("ab")], [2])
        assert clip.calls == []

        clip.transformer.forward_error = None
        backend.score(backend.encode("pq"), [backend.encode("ab")], [2])
        assert len(clip.transformer.forwards) == 2
        assert clip.calls == []

    def test_the_pass_is_pinned_and_under_inference_mode(self):
        torch, clip, backend = self._setup()
        saved = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = False
        try:
            backend.score(backend.encode("pq"), [backend.encode("ab")], [2])
            assert clip.transformer.forwards[0]["pinned"] is True
            assert clip.transformer.forwards[0]["inference"] is True
            assert torch.backends.cudnn.deterministic is False
        finally:
            torch.backends.cudnn.deterministic = saved


class ImageRecordingBackend(StubBackend):
    sees_images = True

    def __init__(self):
        self.prefixes = []

    def encode_prefix(self, text, image=None):
        self.prefixes.append((text, image))
        return self.encode(text), {}


class TestHostImage:
    """Classifier.classify(image=...), the path a host holding a decoded image takes."""

    def _classify(self, state, **kwargs):
        from logit_classifier.classifier import Classifier
        from logit_classifier.config import Config

        backend = ImageRecordingBackend()
        request = parse_request({"state": state, "questions": {"q": {"type": "noul"}}})
        Classifier(Config(), backend).classify(request, **kwargs)
        return backend.prefixes[0]

    def _png_url(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def test_the_image_reaches_encode_prefix_with_the_marker(self):
        host_image = object()
        text, image = self._classify("a cat", image=host_image)
        assert image is host_image
        assert IMAGE_MARKER in text
        assert "a cat" in text

    @pytest.mark.parametrize("state", ["", {}])
    def test_an_empty_state_renders_as_the_service_renders_an_image_only_state(self, state):
        host_text, _ = self._classify(state, image=object())
        service_text, _ = self._classify({"image": self._png_url()})
        assert host_text == service_text

    def test_a_text_only_request_renders_as_before(self):
        text, image = self._classify("a cat")
        assert image is None
        assert IMAGE_MARKER not in text
        assert prefix_content("a cat") in text

    @pytest.mark.parametrize("key", ["image", "screenshot"])
    def test_a_state_image_beside_the_keyword_is_a_conflict(self, key):
        with pytest.raises(SchemaError, match="two images") as caught:
            self._classify({key: self._png_url(), "note": "x"}, image=object())
        assert caught.value.field == f"state.{key}"

    @pytest.mark.parametrize("value", ["", 5, None])
    def test_a_non_image_value_under_an_image_key_is_content_on_both_paths(self, value):
        state = {"image": value, "note": "x"}
        passed = object()
        host_text, host_image = self._classify(state, image=passed)
        service_text, service_image = self._classify(state)
        assert host_image is passed
        assert service_image is None
        assert host_text.replace(f"{IMAGE_MARKER}\n", "") == service_text
