"""Measure the split-choice path, where options exceed one branch's labels.

Each group scores its own members plus an escape label. A group's weight is the
mass it kept off that escape, so groups compete without ever being summarised.
"""

from __future__ import annotations

import numpy as np
from ab_env import FIXTURES  # noqa: F401  sets sys.path

from logit_classifier.backends.hf import HFBackend
from logit_classifier.config import ANSWER_PREFILL, Config
from logit_classifier.prompt import SYSTEM_PROMPT, branch_content, build_branches, prefix_content
from logit_classifier.schema import parse_request
from logit_classifier.scoring import combine_escape, restricted_softmax

LANGUAGES = [
    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani", "Basque",
    "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Burmese", "Catalan", "Cebuano",
    "Chinese", "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian", "Filipino",
    "Finnish", "French", "Galician", "Georgian", "German", "Greek", "Gujarati", "Hausa",
    "Hebrew", "Hindi", "Hungarian", "Icelandic", "Igbo", "Indonesian", "Irish", "Italian",
    "Japanese", "Javanese", "Kannada", "Kazakh", "Khmer", "Korean", "Kurdish", "Kyrgyz",
    "Lao", "Latvian", "Lithuanian", "Luxembourgish", "Macedonian", "Malay", "Malayalam",
    "Maltese", "Marathi", "Mongolian", "Nepali", "Norwegian", "Odia", "Pashto", "Persian",
    "Polish", "Portuguese", "Punjabi", "Romanian", "Russian", "Serbian", "Sindhi",
    "Sinhala", "Slovak", "Slovenian", "Somali", "Spanish", "Sundanese", "Swahili",
    "Swedish", "Tajik", "Tamil", "Telugu", "Thai", "Turkish", "Ukrainian", "Urdu",
    "Uzbek", "Vietnamese", "Welsh", "Xhosa", "Yoruba", "Zulu",
]

SAMPLES = [
    ("Guten Tag, ich habe eine Frage zu meiner Rechnung.", "German"),
    ("Bonjour, je voudrais annuler mon abonnement aujourd'hui.", "French"),
    ("Buenos dias, necesito ayuda con mi cuenta bancaria.", "Spanish"),
    ("Buongiorno, vorrei sapere il prezzo del prodotto.", "Italian"),
    ("Goedemorgen, ik heb een vraag over mijn bestelling.", "Dutch"),
    ("Dzien dobry, mam pytanie dotyczace mojego zamowienia.", "Polish"),
    ("Ola, gostaria de saber o preco deste produto.", "Portuguese"),
    ("Hei, minulla on kysymys tilauksestani.", "Finnish"),
    ("Merhaba, siparisim hakkinda bir sorum var.", "Turkish"),
    ("Xin chao, toi muon hoi ve don hang cua toi.", "Vietnamese"),
]


def run_case(backend: HFBackend, state: str, temperature: float) -> tuple[str, float, float]:
    request = parse_request(
        {"state": state,
         "questions": {"language": {"type": "choice",
                                    "instructions": "What language is this text written in",
                                    "criteria": dict.fromkeys(LANGUAGES)}}}
    )
    branches = build_branches(request.questions)
    prefix_text = backend.render(SYSTEM_PROMPT, prefix_content(request.state), ANSWER_PREFILL,
                                open_ended=True)
    prefix_ids = backend.encode(prefix_text)
    suffixes = [
        backend.encode(backend.render(SYSTEM_PROMPT, branch_content(request.state, b),
                                    ANSWER_PREFILL)[len(prefix_text):])
        for b in branches
    ]
    scored = backend.score(prefix_ids, suffixes, [b.label_count for b in branches])
    probabilities = [restricted_softmax(s.z, temperature=temperature) for s in scored]
    flat = combine_escape(probabilities)[0] if len(branches) > 1 else probabilities[0]
    group_mass = min(s.candidate_mass for s in scored)
    return LANGUAGES[int(flat.argmax())], float(flat.max()), group_mass


def main() -> None:
    backend = HFBackend(Config())
    print(f"{len(LANGUAGES)} options, {len(SAMPLES)} samples\n")
    header = f"{'T':>6} | {'accuracy':>8} | {'mean p(win)':>11} | {'min label mass':>14}"
    print(header)
    print("-" * len(header))

    for temperature in [1.0, 2.5, 6.0]:
        hits, peaks, masses, misses = 0, [], [], []
        for state, truth in SAMPLES:
            got, peak, mass = run_case(backend, state, temperature)
            hits += got == truth
            peaks.append(peak)
            masses.append(mass)
            if got != truth:
                misses.append(f"{truth}->{got}")
        print(f"{temperature:>6.1f} | {hits / len(SAMPLES):>8.3f} | {np.mean(peaks):>11.3f} | {np.mean(masses):>14.3f}"
              + (f"   misses: {', '.join(misses)}" if misses else ""))


if __name__ == "__main__":
    main()
