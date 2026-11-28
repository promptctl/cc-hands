"""The records of a report log, as the tests of what a run cost write them: a test's phases, and the session's start and end."""

import json


def phase(test: str, when: str, duration: float) -> str:
    return json.dumps({"$report_type": "TestReport", "nodeid": test, "when": when, "duration": duration}) + "\n"


def spent(test: str, setup: float, call: float, teardown: float) -> str:
    return phase(test, "setup", setup) + phase(test, "call", call) + phase(test, "teardown", teardown)


STARTED = json.dumps({"$report_type": "SessionStart", "pytest_version": "9.1.1"}) + "\n"
FINISHED = json.dumps({"$report_type": "SessionFinish", "exitstatus": 0}) + "\n"
