"""Calling the HTTP service, using the standard library only.

Start the service in one terminal, then run this in another.

    logit-classifier serve
    uv run python examples/http_client.py
    uv run python examples/http_client.py --diagnostics

The service loads one model and answers one request at a time, so the same request
returns the same numbers whatever else is in flight.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

REQUEST = {
    "state": "I have been trying to connect my Stripe account for 3 days and it keeps failing.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": None,
            },
        },
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
    },
}


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result: dict = json.loads(response.read())
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8077")
    parser.add_argument("--diagnostics", action="store_true",
                        help="also return label mass, branch counts and latency")
    args = parser.parse_args()

    endpoint = f"{args.url}/v1/systemone"
    if args.diagnostics:
        endpoint += "?diagnostics=1"

    try:
        payload = post(endpoint, REQUEST)
    except urllib.error.HTTPError as error:
        # A malformed request comes back as 422 with the offending field named.
        print(f"{error.code}: {error.read().decode('utf-8')}")
        return
    except urllib.error.URLError as error:
        print(f"could not reach {args.url}, is the service running? {error.reason}")
        return

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
