"""Require an explicit integration decision for every public migration."""

from pathlib import Path

import tyrannis.migration as migrations

DEDICATED_INTEGRATIONS = {
    "GlobalBest": "test_global_best.py",
    "IslandMigration": "test_island_migration.py",
}


def test_public_migrations_have_dedicated_integration_coverage() -> None:
    assert set(migrations.__all__) == set(DEDICATED_INTEGRATIONS), (
        "Update dedicated migration integration coverage when public migrations change."
    )
    for filename in DEDICATED_INTEGRATIONS.values():
        assert Path(__file__).with_name(filename).is_file(), filename
