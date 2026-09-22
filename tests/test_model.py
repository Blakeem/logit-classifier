"""Tests that load the real model. Slow, and they need the GPU."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from logit_classifier.backends.base import Backend
from logit_classifier.classifier import Classifier
from logit_classifier.config import ANSWER_PREFILL, Config
from logit_classifier.labels import MAX_LABELS_PER_BRANCH, LabelBoundaryError, verify_label_ids
from logit_classifier.prompt import (
    SYSTEM_PROMPT,
    branch_content,
    build_branches,
    prefix_content,
)
from logit_classifier.schema import parse_request
from logit_classifier.vision import extract_image

FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.model


def _prefix(backend, state, has_image: bool = False) -> str:
    """The span every branch of one request shares, as the backend renders it."""
    return backend.render(SYSTEM_PROMPT, prefix_content(state, has_image), ANSWER_PREFILL,
                          open_ended=True)


def _branch(backend, state, branch, has_image: bool = False) -> str:
    return backend.render(SYSTEM_PROMPT, branch_content(state, branch, has_image), ANSWER_PREFILL)


@pytest.fixture(scope="module")
def classifier(tmp_path_factory) -> Classifier:
    config = Config(calibration_path=tmp_path_factory.mktemp("cal") / "calibration.json")
    return Classifier(config)


@pytest.fixture(scope="module")
def tokenizer(classifier: Classifier):
    return classifier.backend.tokenizer


class TestReadoutPort:
    def test_the_local_backend_satisfies_the_port(self, classifier):
        assert isinstance(classifier.backend, Backend)


class TestLabelBoundary:
    def test_all_labels_are_one_token_after_the_prefill(self, tokenizer):
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True
        )
        ids = verify_label_ids(tokenizer, base + ANSWER_PREFILL, MAX_LABELS_PER_BRANCH)
        assert len(set(ids)) == MAX_LABELS_PER_BRANCH

    def test_a_trailing_space_prefill_is_rejected(self, tokenizer):
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True
        )
        with pytest.raises(LabelBoundaryError):
            verify_label_ids(tokenizer, base + "Answer: ", MAX_LABELS_PER_BRANCH)


class TestPrefixSplit:
    def test_prefix_and_suffix_tokenise_independently(self, classifier, tokenizer):
        request = parse_request(
            json.loads((FIXTURES / "quickstart_request.json").read_text(encoding="utf-8"))
        )
        prefix_text = _prefix(classifier.backend, request.state)
        prefix_ids = classifier.backend.encode(prefix_text)
        for branch in build_branches(request.questions):
            full = _branch(classifier.backend, request.state, branch)
            suffix_ids = classifier.backend.encode(full[len(prefix_text):])
            assert prefix_ids + suffix_ids == classifier.backend.encode(full)


class TestEndToEnd:
    def test_published_jev_request_returns_the_published_choices(self, classifier):
        request = parse_request(
            json.loads((FIXTURES / "quickstart_request.json").read_text(encoding="utf-8"))
        )
        response, _ = classifier.classify(request)
        answers = response.answers
        assert answers["department"].choice == "technical"
        assert round(answers["frustration"].score) == 1
        assert answers["is_urgent"].noul > 0.5

    def test_probabilities_form_a_distribution(self, classifier):
        request = parse_request(
            json.loads((FIXTURES / "quickstart_request.json").read_text(encoding="utf-8"))
        )
        response, _ = classifier.classify(request)
        assert sum(response.answers["department"].probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert sum(response.answers["frustration"].probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    def test_answer_keys_mirror_question_keys(self, classifier):
        request = parse_request(
            json.loads((FIXTURES / "quickstart_request.json").read_text(encoding="utf-8"))
        )
        response, _ = classifier.classify(request)
        assert set(response.answers) == set(request.questions)

    def test_split_choice_covers_every_option(self, classifier):
        criteria = {f"label_{i:03d}": None for i in range(120)}
        request = parse_request(
            {"state": "some text", "questions": {"q": {"type": "choice", "criteria": criteria}}}
        )
        response, diagnostics = classifier.classify(request)
        probabilities = response.answers["q"].probabilities
        assert set(probabilities) == set(criteria)
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert diagnostics.branch_counts["q"] == 3

    def test_repeated_requests_agree(self, classifier):
        request = parse_request(
            json.loads((FIXTURES / "quickstart_request.json").read_text(encoding="utf-8"))
        )
        first, _ = classifier.classify(request)
        second, _ = classifier.classify(request)
        assert first.answers["department"].choice == second.answers["department"].choice

    def test_label_mass_is_high_on_a_well_formed_question(self, classifier):
        request = parse_request(
            json.loads((FIXTURES / "quickstart_request.json").read_text(encoding="utf-8"))
        )
        _, diagnostics = classifier.classify(request)
        every = [m for masses in diagnostics.candidate_mass.values() for m in masses]
        assert min(every) > 0.5


class TestDeterminism:
    def test_identical_request_gives_identical_logits(self, classifier):
        request = parse_request(
            {"state": "The build fails on Windows only.",
             "questions": {"q": {"type": "choice", "criteria": {"bug": None, "question": None, "praise": None}}}}
        )
        branches = build_branches(request.questions)
        prefix_text = _prefix(classifier.backend, request.state)
        prefix_ids = classifier.backend.encode(prefix_text)
        suffix_ids = [
            classifier.backend.encode(_branch(classifier.backend, request.state, b)[len(prefix_text):])
            for b in branches
        ]
        counts = [b.label_count for b in branches]
        first = classifier.backend.score(prefix_ids, suffix_ids, counts)
        second = classifier.backend.score(prefix_ids, suffix_ids, counts)
        assert np.array_equal(first[0].z, second[0].z)

class TestImages:
    def _swatch(self, colour):
        image = Image.new("RGB", (224, 224), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 184, 184), fill=colour)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def _ask(self, classifier, colour):
        return parse_request(
            {"state": {"image": self._swatch(colour)},
             "questions": {"q": {"type": "choice",
                                 "instructions": "What colour is the square",
                                 "criteria": {"red": None, "green": None, "blue": None}}}}
        )

    def test_the_model_reads_an_image(self, classifier):
        if not classifier.backend.sees_images:
            pytest.skip("configured model has no vision tower")
        for colour in ("red", "green", "blue"):
            answer = classifier.classify(self._ask(classifier, colour))[0].answers["q"]
            assert answer.choice == colour

    def test_the_answer_letters_still_hold_every_bit_of_mass(self, classifier):
        if not classifier.backend.sees_images:
            pytest.skip("configured model has no vision tower")
        diagnostics = classifier.classify(self._ask(classifier, "red"))[1]
        assert min(diagnostics.candidate_mass["q"]) > 0.9

    def test_the_image_expands_inside_the_prefix_without_moving_the_boundary(self, classifier):
        if not classifier.backend.sees_images:
            pytest.skip("configured model has no vision tower")
        request = self._ask(classifier, "green")
        state, image = extract_image(request.state)
        backend = classifier.backend
        prefix_text = _prefix(backend, state, True)
        prefix_ids, vision = backend.encode_prefix(prefix_text, image)
        branch = build_branches(request.questions, "joint", classifier.config.abstain)[0]
        full_text = _branch(backend, state, branch, True)
        suffix_ids = backend.encode(full_text[len(prefix_text):])
        joined, _ = backend.encode_prefix(full_text, image)
        assert prefix_ids + suffix_ids == joined
        assert "pixel_values" in vision and "mm_token_type_ids" in vision
