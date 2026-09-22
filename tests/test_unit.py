"""Unit tests that need no model weights."""

from __future__ import annotations

import ast
import base64
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
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


class TestCoreIsolation:
    def test_importing_the_package_loads_no_host(self):
        # Prime directive 4. A ComfyUI pack imports this package beside ComfyUI's own
        # pinned CUDA build, so a stray top-level torch import would be a hard break.
        # A subprocess is the only honest check, since pytest has already imported PIL.
        probe = (
            "import sys; import logit_classifier; "
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
        from logit_classifier.backends.hf import _determinism

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
        from logit_classifier.backends.hf import _determinism

        torch_host.backends.cuda.matmul.fp32_precision = "none"
        torch_host.backends.mkldnn.matmul.fp32_precision = "none"
        torch_host.backends.cudnn.deterministic = False
        host = self._snapshot(torch_host)

        with pytest.raises(RuntimeError), _determinism():
            raise RuntimeError("the forward pass blew up")

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
                     "LogitClassifierError", "COMFY_SOCKET_TYPE"):
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
