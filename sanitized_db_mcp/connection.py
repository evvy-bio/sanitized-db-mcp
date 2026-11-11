"""PostgreSQL access for the query service."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import os

import psycopg
from psycopg.conninfo import conninfo_to_dict

from .allowlist import Allowlist
from .errors import ConfigurationError, QueryExecutionError, QueryTimeoutError


logger = logging.getLogger(__name__)
DATABASE_ROLE = "agent_readonly"

_ROLE_QUERY = """
SELECT session_user, current_user,
       rolsuper OR rolcreaterole OR rolcreatedb OR rolreplication OR rolbypassrls
FROM pg_catalog.pg_roles
WHERE rolname = current_user
"""

# Ask PostgreSQL for effective SELECT privileges, including table-wide,
# inherited and PUBLIC grants. The system catalog schemas serve metadata.
_READ_GRANTS_QUERY = """
SELECT namespace.nspname, relation.relname, attribute.attname
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
LEFT JOIN pg_catalog.pg_attribute AS attribute
    ON attribute.attrelid = relation.oid
    AND attribute.attnum <> 0
    AND NOT attribute.attisdropped
WHERE relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
    AND namespace.nspname NOT IN ('pg_catalog', 'information_schema')
    AND namespace.nspname !~ '^pg_(temp|toast)(_|$)'
    AND CASE
        WHEN relation.relkind = 'S' THEN
            pg_catalog.has_sequence_privilege(relation.oid, 'SELECT')
        WHEN attribute.attnum IS NULL THEN
            pg_catalog.has_table_privilege(relation.oid, 'SELECT')
        ELSE
            pg_catalog.has_column_privilege(relation.oid, attribute.attnum, 'SELECT')
    END
"""


def get_connection_string() -> str:
    """Require an explicit DSN for the query service."""
    if os.environ.get("RENDER_POSTGRES_ID") and os.environ.get("RENDER_API_KEY"):
        raise ConfigurationError(
            "Remove Render database discovery settings and configure DATABASE_URL"
        )
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ConfigurationError("DATABASE_URL is required for the query service")
    return database_url


@contextmanager
def _verified_connection(
    allowlist: Allowlist, connection_string: str | None
) -> Iterator[psycopg.Connection]:
    if connection_string is None:
        connection_string = get_connection_string()
    try:
        settings = conninfo_to_dict(connection_string)
    except psycopg.Error:
        raise ConfigurationError("Invalid DATABASE_URL") from None
    if settings.get("user") != DATABASE_ROLE:
        raise ConfigurationError(f"DATABASE_URL must specify the {DATABASE_ROLE} login")

    hosts = (settings.get("hostaddr") or settings.get("host", "")).split(",")
    connection_options = {"autocommit": True, "connect_timeout": 5}
    if any(host not in {"localhost", "127.0.0.1", "::1"} for host in hosts):
        if settings.get("sslmode") not in {"verify-ca", "verify-full"}:
            connection_options["sslmode"] = "require"

    with psycopg.connect(connection_string, **connection_options) as conn:
        conn.read_only = True
        with conn.transaction():
            conn.execute("SET LOCAL statement_timeout = '5s'")
            conn.execute("SET LOCAL search_path = pg_catalog, public, pg_temp")

            if conn.execute(_ROLE_QUERY).fetchone() != (DATABASE_ROLE, DATABASE_ROLE, False):
                raise ConfigurationError(
                    f"Database session must use the dedicated {DATABASE_ROLE} login"
                )

            readable = set(conn.execute(_READ_GRANTS_QUERY).fetchall())
            if readable - allowlist.database_columns():
                raise ConfigurationError("Database SELECT grants exceed the approved allowlist")

            yield conn


def verify_database_access(
    allowlist: Allowlist, connection_string: str | None = None
) -> None:
    """Check the configured login and read grants without issuing a tool query."""
    try:
        with _verified_connection(allowlist, connection_string):
            pass
    except psycopg.Error as error:
        raise ConfigurationError(
            f"Database access could not be verified ({type(error).__name__})"
        ) from None


def execute_query(
    sql: str, connection_string: str | None = None, *, allowlist: Allowlist
) -> list[dict]:
    """Execute sanitized SQL on the same read-only connection that passed the grant check."""
    try:
        with _verified_connection(allowlist, connection_string) as conn, conn.cursor() as cur:
            cur.execute(sql)

            if cur.description is None:
                return []

            columns = [desc.name for desc in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    except psycopg.errors.QueryCanceled:
        raise QueryTimeoutError("Query exceeded 5-second timeout") from None
    except psycopg.errors.ReadOnlySqlTransaction:
        raise QueryExecutionError("Write operations are not permitted") from None
    except psycopg.errors.InsufficientPrivilege:
        raise QueryExecutionError("Insufficient privileges for this query") from None
    except psycopg.Error as error:
        logger.error("Database error: %s", type(error).__name__)
        raise QueryExecutionError(f"Database error: {type(error).__name__}") from None
