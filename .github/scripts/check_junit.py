"""Require a nonempty, fully successful pytest JUnit report for optional jobs."""

import sys
from pathlib import Path
from xml.etree import ElementTree


def check_report(path: Path) -> int:
    """Return the executed case count; reject missing or unsuccessful reports."""
    root = ElementTree.parse(path).getroot()
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError(f"Unexpected JUnit root: {root.tag}")

    suites = list(root.iter("testsuite"))
    cases = list(root.iter("testcase"))
    if not suites or not cases:
        raise ValueError("JUnit report contains no executed tests")

    # Check both counters and elements, including pytest's collection skips.
    for node in root.iter():
        if node.tag in {"skipped", "failure", "error"}:
            raise ValueError(f"JUnit report contains {node.tag}: {node.attrib}")
        if node.tag in {"testsuite", "testsuites"}:
            for field in ("skipped", "failures", "errors"):
                count = node.get(field)
                if count is None and node.tag == "testsuite":
                    raise ValueError(f"JUnit suite is missing {field}")
                if count is not None and int(count) != 0:
                    raise ValueError(f"JUnit report records {field}={count}")
            declared = node.get("tests")
            if declared is None and node.tag == "testsuite":
                raise ValueError("JUnit suite is missing its test count")
            if declared is not None and int(declared) != len(
                list(node.iter("testcase"))
            ):
                raise ValueError("JUnit test count does not match its cases")

    return len(cases)


def main() -> None:
    """Validate the single report supplied by the workflow."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_junit.py REPORT.xml")
    path = Path(sys.argv[1])
    try:
        count = check_report(path)
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        raise SystemExit(f"Invalid optional-test report {path}: {exc}") from exc
    print(f"Verified {count} executed tests with no skips, failures, or errors")


if __name__ == "__main__":
    main()
