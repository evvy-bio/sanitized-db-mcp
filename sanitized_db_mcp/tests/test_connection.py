"""Synthetic tests for the database login and effective-grant boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import psycopg
import pytest

from sanitized_db_mcp.allowlist import Allowlist
from sanitized_db_mcp.connection import execute_query, verify_database_access
from sanitized_db_mcp.errors import ConfigurationError


_DSN = "host=127.0.0.1 dbname=synthetic user=agent_readonly"
_QUERY = "SELECT item_id FROM example_items"


@pytest.fixture
def approved_allowlist():
    return Allowlist(
        {
            "example_items": {"item_id": {}},
            "reporting.totals": {"label": {}},
        },
        set(),
    )


@pytest.fixture
def database(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    metadata = connection.execute.return_value
    metadata.fetchone.return_value = ("agent_readonly", "agent_readonly", False)
    metadata.fetchall.return_value = [
        ("public", "example_items", "item_id"),
        ("reporting", "totals", "label"),
    ]
    connect = Mock(return_value=connection)
    monkeypatch.setattr(psycopg, "connect", connect)
    return SimpleNamespace(connection=connection, metadata=metadata, connect=connect)


def test_readiness_checks_approved_grants_without_executing_a_tool_query(
    approved_allowlist, database
):
    verify_database_access(approved_allowlist, _DSN)

    database.connection.cursor.assert_not_called()


def test_query_uses_the_verified_read_only_transaction(approved_allowlist, database):
    cursor = database.connection.cursor.return_value.__enter__.return_value
    cursor.description = [SimpleNamespace(name="item_id")]
    cursor.fetchall.return_value = [(17,)]

    rows = execute_query(_QUERY, _DSN, allowlist=approved_allowlist)

    assert rows == [{"item_id": 17}]
    database.connect.assert_called_once()
    assert database.connection.read_only is True
    database.connection.transaction.assert_called_once_with()
    cursor.execute.assert_called_once_with(_QUERY)


@pytest.mark.parametrize("configured_user", ["", "other_login"])
def test_dsn_must_explicitly_name_the_dedicated_login(
    configured_user, approved_allowlist, database
):
    dsn = "host=127.0.0.1 dbname=synthetic"
    if configured_user:
        dsn += f" user={configured_user}"

    with pytest.raises(ConfigurationError, match="must specify the agent_readonly login"):
        verify_database_access(approved_allowlist, dsn)

    database.connect.assert_not_called()


@pytest.mark.parametrize(
    "identity",
    [
        ("other_login", "agent_readonly", False),
        ("agent_readonly", "other_role", False),
        ("agent_readonly", "agent_readonly", True),
        None,
    ],
)
def test_refuses_unverified_or_elevated_session_identity(
    identity, approved_allowlist, database
):
    database.metadata.fetchone.return_value = identity

    with pytest.raises(ConfigurationError, match="dedicated agent_readonly login"):
        execute_query(_QUERY, _DSN, allowlist=approved_allowlist)

    database.metadata.fetchall.assert_not_called()
    database.connection.cursor.assert_not_called()


@pytest.mark.parametrize(
    "extra_grant",
    [
        ("public", "example_items", "internal_note"),
        ("public", "example_items", "ITEM_ID"),
        ("reporting", "example_items", "item_id"),
        ("public", "unapproved_table", "item_id"),
        ("public", "empty_table", None),
    ],
)
def test_excess_effective_grants_fail_readiness(extra_grant, approved_allowlist, database):
    database.metadata.fetchall.return_value.append(extra_grant)

    with pytest.raises(ConfigurationError, match="SELECT grants exceed the approved allowlist"):
        verify_database_access(approved_allowlist, _DSN)


def test_excess_grants_stop_tool_sql_before_execution(approved_allowlist, database):
    database.metadata.fetchall.return_value.append(("public", "example_items", "internal_note"))

    with pytest.raises(ConfigurationError, match="SELECT grants exceed the approved allowlist"):
        execute_query(_QUERY, _DSN, allowlist=approved_allowlist)

    database.connection.cursor.assert_not_called()


def test_conflicting_render_discovery_settings_fail_before_connecting(
    monkeypatch, approved_allowlist, database
):
    monkeypatch.setenv("DATABASE_URL", _DSN)
    monkeypatch.setenv("RENDER_POSTGRES_ID", "synthetic-instance")
    monkeypatch.setenv("RENDER_API_KEY", "synthetic-value")

    with pytest.raises(ConfigurationError, match="Remove Render database discovery"):
        verify_database_access(approved_allowlist)

    database.connect.assert_not_called()


def test_missing_database_url_fails_readiness(monkeypatch, approved_allowlist, database):
    for name in ("DATABASE_URL", "RENDER_POSTGRES_ID", "RENDER_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigurationError, match="DATABASE_URL is required"):
        verify_database_access(approved_allowlist)

    database.connect.assert_not_called()


def test_driver_failure_keeps_readiness_error_generic(approved_allowlist, database):
    database.connect.side_effect = psycopg.OperationalError("synthetic private diagnostic")

    with pytest.raises(ConfigurationError, match="Database access could not be verified") as error:
        verify_database_access(approved_allowlist, _DSN)

    assert "synthetic private diagnostic" not in str(error.value)
    assert error.value.__cause__ is None
