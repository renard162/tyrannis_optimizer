"""Prepare the documentation site and guard Pages deployments."""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen


def identity() -> dict[str, object]:
    return {
        "commit_sha": os.environ["GITHUB_SHA"],
        "actions_run_id": os.environ["GITHUB_RUN_ID"],
        "actions_run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
    }


def mapping(value: object) -> dict[str, object]:
    """Narrow JSON objects at the API decoder boundary."""
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return cast(dict[str, object], value)


def api(path: str) -> dict[str, object]:
    request = Request(
        f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return mapping(json.load(response))


def objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("Expected an API array")
    return [mapping(item) for item in value]


def deploy_allowed(
    metadata: dict[str, object],
    run: dict[str, object],
    head: str,
    latest: dict[str, object],
    event: str,
    ref: str,
) -> bool:
    return (
        event == "push"
        and ref == "refs/heads/master"
        and run.get("event") == "push"
        and run.get("head_branch") == "master"
        and head == run.get("head_sha") == metadata["commit_sha"]
        and str(run.get("id")) == metadata["actions_run_id"]
        and str(run.get("run_attempt")) == metadata["actions_run_attempt"]
        and latest.get("id") == run.get("id")
        and run.get("conclusion") != "cancelled"
    )


def guard(metadata: dict[str, object]) -> bool:
    run = api(f"actions/runs/{metadata['actions_run_id']}")
    ref = api("git/ref/heads/master")
    runs = objects(
        api(
            f"actions/workflows/{run['workflow_id']}/runs?branch=master&event=push&per_page=1"
        )["workflow_runs"]
    )
    return bool(runs) and deploy_allowed(
        metadata,
        run,
        str(mapping(ref["object"])["sha"]),
        runs[0],
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REF"],
    )


def main() -> None:
    command = sys.argv[1]
    if command == "prepare":
        # Replace this source with the future documentation build, in the same job.
        _ = shutil.copytree(".github/pages", "site")
    elif command == "guard":
        allowed = guard(identity())
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            _ = output.write(f"allowed={str(allowed).lower()}\n")
        print(
            "Deploy authorized" if allowed else "Obsolete/cancelled run: deploy skipped"
        )
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
