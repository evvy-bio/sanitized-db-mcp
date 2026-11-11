"""Relation and column authorization with a synthetic reporting schema."""

import json
from unittest.mock import Mock

import anyio
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from pglast import ast, parse_sql
import pytest

from sanitized_db_mcp import server
from sanitized_db_mcp.allowlist import Allowlist
from sanitized_db_mcp.errors import RestrictedColumnError
from sanitized_db_mcp.sanitizer import sanitize_query


@pytest.fixture
def reporting_allowlist():
    return Allowlist(
        tables={
            "orders": {"order_id": {}, "account_id": {}, "state": {}},
            "accounts": {"account_id": {}, "region": {}},
            "reporting.daily_orders": {"order_id": {}},
            "CaseSensitive": {"OrderID": {}},
        },
        allowed_functions={"COUNT", "SUM", "ARRAY_AGG", "ROW_NUMBER", "GENERATE_SERIES"},
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM archived_orders",
        "WITH daily AS (SELECT count(*) AS total FROM archived_orders) SELECT total FROM daily",
        "SELECT order_id FROM orders UNION ALL SELECT order_id FROM archived_orders",
        "SELECT order_id FROM archived_orders INTERSECT SELECT order_id FROM orders",
        "SELECT order_id FROM orders EXCEPT SELECT order_id FROM archived_orders",
        "SELECT o.order_id FROM orders o JOIN (SELECT order_id FROM archived_orders) a "
        "ON o.order_id = a.order_id",
        "SELECT o.order_id FROM orders o LEFT JOIN LATERAL "
        "(SELECT count(*) AS total FROM archived_orders) a ON true",
        "SELECT count(*) FROM archived_orders TABLESAMPLE SYSTEM (5)",
        "SELECT o.order_id FROM orders o JOIN archived_orders TABLESAMPLE SYSTEM (5) ON true",
        "SELECT v.total FROM (VALUES ((SELECT count(*) FROM archived_orders))) v(total)",
        "SELECT order_id FROM orders LIMIT (SELECT count(*) FROM archived_orders)",
        "SELECT order_id FROM orders WHERE EXISTS (SELECT 1 FROM archived_orders)",
        "SELECT order_id FROM other.orders",
        'SELECT order_id FROM "Orders"',
    ],
)
def test_checks_relation_in_each_scope(reporting_allowlist, sql):
    with pytest.raises(RestrictedColumnError, match="not in the allowlist"):
        sanitize_query(sql, reporting_allowlist)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT order_id",
        "SELECT report.order_id FROM orders o",
        "SELECT report.* FROM orders o",
        "SELECT o FROM orders o",
        "SELECT count(o.*) FROM orders o",
        "SELECT order_id FROM orders o JOIN accounts a ON o.account_id = a.account_id",
        "SELECT account_id FROM orders o JOIN accounts a ON o.account_id = a.account_id",
        "SELECT d.order_id FROM (SELECT state FROM orders) d",
        "SELECT d.order_id FROM (SELECT order_id, account_id AS order_id FROM orders) d",
        "SELECT d.* FROM (SELECT order_id, account_id AS order_id FROM orders) d",
        "WITH report AS (SELECT order_id FROM orders) SELECT report.state FROM report",
        "SELECT d.order_id FROM (SELECT order_id + 1 FROM orders) d",
        "SELECT o.order_id FROM orders o JOIN accounts a USING (account_id)",
        "SELECT o.order_id FROM orders o NATURAL JOIN accounts a",
        "SELECT o.order_id FROM orders AS o(order_id)",
        "SELECT d.account_id FROM orders o, (SELECT o.account_id) d",
        "SELECT d.account_id FROM LATERAL (SELECT o.account_id) d, orders o",
        "SELECT o.order_id FROM orders o, accounts a "
        "JOIN (VALUES (7)) d(account_id) ON o.account_id = d.account_id",
        "SELECT o.order_id FROM orders o WHERE EXISTS "
        "(SELECT 1 FROM accounts o WHERE o.order_id = 101)",
        'SELECT "OrderID" FROM "CaseSensitive" WHERE orderid = 1',
    ],
)
def test_rejects_column_without_proven_source(reporting_allowlist, sql):
    with pytest.raises(RestrictedColumnError):
        sanitize_query(sql, reporting_allowlist)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT order_id FROM orders WHERE note = 'draft'",
        "SELECT order_id FROM orders ORDER BY note",
        "SELECT count(*) FROM orders GROUP BY note",
        "SELECT DISTINCT ON (note) order_id FROM orders",
        "SELECT count(*) FROM orders HAVING count(note) > 0",
        "SELECT array_agg(order_id ORDER BY note) FROM orders",
        "SELECT count(*) FILTER (WHERE note IS NOT NULL) FROM orders",
        "SELECT row_number() OVER (ORDER BY note) FROM orders",
        "SELECT row_number() OVER w FROM orders WINDOW w AS (ORDER BY note)",
        "SELECT o.note IN (SELECT a.region FROM accounts a) FROM orders o",
        "WITH report AS (SELECT note FROM orders) SELECT 1 FROM report",
        "SELECT 1 FROM orders o WHERE EXISTS (SELECT o.note)",
    ],
)
def test_requires_approved_column_in_non_redactable_expressions(reporting_allowlist, sql):
    with pytest.raises(RestrictedColumnError, match="restricted in this expression"):
        sanitize_query(sql, reporting_allowlist)


@pytest.mark.parametrize(
    ("sql", "tables"),
    [
        (
            "WITH recent(id) AS (SELECT order_id FROM orders), "
            "report AS (SELECT id FROM recent) SELECT report.id FROM report ORDER BY id",
            ["orders"],
        ),
        (
            "WITH selected AS (SELECT order_id FROM orders), "
            "locations AS (SELECT region FROM accounts) "
            "SELECT order_id, region FROM selected CROSS JOIN locations",
            ["orders", "accounts"],
        ),
        (
            "WITH orders AS (SELECT account_id AS id FROM accounts) SELECT orders.id FROM orders",
            ["accounts"],
        ),
        (
            "WITH orders AS (SELECT account_id AS id FROM accounts) "
            "SELECT public.orders.order_id FROM public.orders",
            ["accounts", "orders"],
        ),
        (
            "SELECT o.order_id, d.region FROM orders o "
            "JOIN (SELECT account_id, region FROM accounts) d ON d.account_id = o.account_id",
            ["orders", "accounts"],
        ),
        (
            "SELECT o.order_id, d.region FROM orders o LEFT JOIN LATERAL "
            "(SELECT a.region FROM accounts a WHERE a.account_id = o.account_id) d ON true",
            ["orders", "accounts"],
        ),
        (
            "SELECT o.order_id FROM orders o WHERE EXISTS "
            "(SELECT 1 FROM accounts a WHERE a.account_id = o.account_id)",
            ["orders", "accounts"],
        ),
        (
            "SELECT d.id FROM (SELECT order_id AS id FROM orders UNION ALL "
            "SELECT order_id AS another_name FROM reporting.daily_orders) d ORDER BY id",
            ["orders", "reporting.daily_orders"],
        ),
        ('SELECT "CaseSensitive"."OrderID" FROM "CaseSensitive"', ["CaseSensitive"]),
        ("SELECT v.number FROM (VALUES (1), (2)) AS v(number)", []),
        ("SELECT s.number FROM generate_series(1, 2) AS s(number)", []),
        ("SELECT order_id FROM orders TABLESAMPLE BERNOULLI (100) REPEATABLE (7)", ["orders"]),
    ],
)
def test_resolves_validated_sources(reporting_allowlist, sql, tables):
    result = sanitize_query(sql, reporting_allowlist)
    assert result.tables_accessed == tables
    assert not result.was_rewritten


def test_physical_relation_identity_is_pinned(reporting_allowlist):
    result = sanitize_query("SELECT order_id FROM orders", reporting_allowlist)
    relation = parse_sql(result.rewritten_sql)[0].stmt.fromClause[0]

    assert (relation.schemaname, relation.relname) == ("public", "orders")


def test_sampling_statement_round_trips(reporting_allowlist):
    result = sanitize_query(
        "SELECT order_id FROM orders TABLESAMPLE pg_catalog.bernoulli (100) REPEATABLE (7)",
        reporting_allowlist,
    )
    sample = parse_sql(result.rewritten_sql)[0].stmt.fromClause[0]

    assert isinstance(sample, ast.RangeTableSample)
    assert sample.relation.schemaname == "public"
    assert tuple(part.sval for part in sample.method) == ("bernoulli",)


def test_derived_star_only_exposes_validated_outputs(reporting_allowlist):
    result = sanitize_query(
        "SELECT d.* FROM (SELECT o.* FROM orders o) d",
        reporting_allowlist,
    )
    outer = parse_sql(result.rewritten_sql)[0].stmt
    inner = outer.fromClause[0].subquery

    assert result.was_rewritten
    assert len(inner.targetList) == len(outer.targetList) == 3
    assert [target.val.fields[-1].sval for target in inner.targetList] == [
        "account_id", "order_id", "state",
    ]
    assert [target.val.fields[-1].sval for target in outer.targetList] == [
        "account_id", "order_id", "state",
    ]


def test_explicit_hidden_outer_projection_is_a_literal(reporting_allowlist):
    result = sanitize_query('SELECT o.note AS "report; note" FROM orders o', reporting_allowlist)
    statements = parse_sql(result.rewritten_sql)
    target = statements[0].stmt.targetList[0]

    assert len(statements) == 1
    assert result.columns_redacted == ["orders.note"]
    assert target.name == "report; note"
    assert isinstance(target.val, ast.A_Const)
    assert target.val.val.sval == "[REDACTED]"


@pytest.mark.asyncio
async def test_mcp_checks_scope_before_execution(tmp_path, monkeypatch):
    policy = tmp_path / "reporting.yaml"
    policy.write_text("tables: {orders: {columns: {order_id: {}}}}\nallowed_functions: [COUNT]\n")
    monkeypatch.setenv("ALLOWLIST_PATH", str(policy))
    executor = Mock(return_value=[{"order_id": 101}])
    monkeypatch.setattr(server, "execute_query", executor)
    mcp_server, allowlist = server.create_server()

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(mcp_server.run, *server_streams, mcp_server.create_initialization_options())
            async with ClientSession(*client_streams, read_timeout_seconds=5) as client:
                await client.initialize()
                for sql in (
                    "WITH daily AS (SELECT count(*) AS total FROM archived_orders) SELECT total FROM daily",
                    "SELECT d.order_id FROM (SELECT 1 AS total) d",
                ):
                    result = await client.call_tool("query", {"sql": sql})
                    assert result.content[0].text == "Error: Query references restricted columns or tables"
                    executor.assert_not_called()

                result = await client.call_tool("query", {"sql": "SELECT order_id FROM orders"})
                _, rows = result.content[0].text.split("\n", 1)
                assert json.loads(rows) == [{"order_id": 101}]
                executor.assert_called_once_with("SELECT order_id FROM public.orders", allowlist=allowlist)
            tasks.cancel_scope.cancel()
