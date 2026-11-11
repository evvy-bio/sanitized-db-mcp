"""Validate and rewrite PostgreSQL SELECT queries before execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from pglast import ast, parse_sql
from pglast.enums import A_Expr_Kind, SetOperation
from pglast.parser import ParseError
from pglast.stream import RawStream

from .allowlist import Allowlist
from .errors import (
    DisallowedFunctionError,
    QuerySyntaxError,
    RestrictedColumnError,
    StatementTypeError,
    SystemCatalogError,
)

# System schemas that must never be accessed
BLOCKED_SCHEMAS = frozenset(
    {
        "pg_catalog",
        "information_schema",
        "pg_temp",
        "pg_toast",
    }
)

# System table prefixes to block
BLOCKED_TABLE_PREFIXES = ("pg_", "sql_")

# Functions that can dump entire rows or exfiltrate data
# (blocked even if they appear to take allowlisted args)
ALWAYS_BLOCKED_FUNCTIONS = frozenset(
    {
        "ROW_TO_JSON",
        "TO_JSON",
        "TO_JSONB",
        "JSON_AGG",
        "JSONB_AGG",
        "JSON_OBJECT_AGG",
        "JSONB_OBJECT_AGG",
        "JSON_BUILD_OBJECT",
        "JSONB_BUILD_OBJECT",
        "JSON_POPULATE_RECORD",
        "JSONB_POPULATE_RECORD",
        "JSON_POPULATE_RECORDSET",
        "JSONB_POPULATE_RECORDSET",
        "XMLELEMENT",
        "XMLFOREST",
        "XMLAGG",
        "QUERY_TO_XML",
        "TABLE_TO_XML",
        "CURSOR_TO_XML",
        "DBLINK",
        "DBLINK_EXEC",
        "PG_SLEEP",
        "PG_ADVISORY_LOCK",
        "PG_ADVISORY_UNLOCK",
        "SET_CONFIG",
        "CURRENT_SETTING",
        "HAS_TABLE_PRIVILEGE",
        "HAS_COLUMN_PRIVILEGE",
        "HAS_SCHEMA_PRIVILEGE",
        "HAS_DATABASE_PRIVILEGE",
        "PG_READ_FILE",
        "PG_READ_BINARY_FILE",
        "PG_LS_DIR",
        "PG_STAT_FILE",
        "ENCODE",
        "DECODE",
        "MD5",
        "SHA256",
        "SHA512",
        "DIGEST",
        "CRYPT",
        "LO_IMPORT",
        "LO_EXPORT",
        "LO_GET",
        "COPY",
    }
)


@dataclass
class SanitizeResult:
    """Result of sanitizing a SQL query."""

    original_sql: str
    rewritten_sql: str
    tables_accessed: list[str] = field(default_factory=list)
    columns_accessed: list[str] = field(default_factory=list)
    columns_redacted: list[str] = field(default_factory=list)
    was_rewritten: bool = False


def sanitize_query(sql: str, allowlist: Allowlist) -> SanitizeResult:
    """Return SQL with verified column sources and redacted outer projections."""
    try:
        stmts = parse_sql(sql)
    except ParseError as error:
        raise QuerySyntaxError(f"Parse error: {error}") from error

    if not stmts:
        raise QuerySyntaxError("Empty query")
    if len(stmts) != 1:
        raise StatementTypeError("Multiple statements are not allowed")

    result = SanitizeResult(original_sql=sql, rewritten_sql="")
    _QueryValidator(allowlist, result).select(stmts[0].stmt, {}, redact=True)
    result.rewritten_sql = RawStream()(stmts[0])
    result.tables_accessed = list(dict.fromkeys(result.tables_accessed))
    result.columns_accessed = list(dict.fromkeys(result.columns_accessed))
    result.columns_redacted = list(dict.fromkeys(result.columns_redacted))
    return result


# Unknown labels occur on computed outputs without an explicit alias.
_Output = tuple[str | None, ...]
_CTEs = dict[str, _Output]


@dataclass(frozen=True)
class _Source:
    qualifier: tuple[str, ...]
    columns: _Output
    table: str | None = None

    @property
    def name(self) -> str:
        return self.qualifier[-1]

    def may_have_column(self, column: str) -> bool:
        # The policy has no inventory of hidden physical columns.
        return self.table is not None or column in self.columns or None in self.columns


_Scope = tuple[_Source, ...]
_Scopes = tuple[_Scope, ...]


class _QueryValidator:
    def __init__(self, allowlist: Allowlist, result: SanitizeResult):
        self.allowlist = allowlist
        self.result = result

    def select(
        self,
        stmt: ast.Node,
        ctes: _CTEs,
        outer: _Scopes = (),
        *,
        redact: bool = False,
    ) -> _Output:
        if not isinstance(stmt, ast.SelectStmt):
            raise StatementTypeError(f"Only SELECT statements are allowed, got {type(stmt).__name__}")
        if stmt.intoClause or stmt.lockingClause:
            raise StatementTypeError("SELECT INTO and locking clauses are not allowed")

        if stmt.withClause:
            if stmt.withClause.recursive:
                raise RestrictedColumnError("Recursive CTE output cannot be resolved")
            ctes = ctes.copy()
            names = set()
            for cte in stmt.withClause.ctes:
                if cte.ctename in names:
                    raise RestrictedColumnError("Duplicate CTE name")
                columns = self.select(cte.ctequery, ctes, outer)
                ctes[cte.ctename] = _rename_columns(columns, cte.aliascolnames)
                names.add(cte.ctename)

        sources: _Scope = ()
        for item in stmt.fromClause or ():
            sources += self._from_item(item, ctes, outer, sources)
        scopes = (sources, *outer)

        if stmt.op != SetOperation.SETOP_NONE:
            columns = self.select(stmt.larg, ctes, outer)
            if len(columns) != len(self.select(stmt.rarg, ctes, outer)):
                raise RestrictedColumnError("Set operation inputs have different column counts")
        elif stmt.valuesLists:
            width = len(stmt.valuesLists[0])
            if any(len(row) != width for row in stmt.valuesLists):
                raise RestrictedColumnError("VALUES rows have different column counts")
            self._expression(stmt.valuesLists, scopes, ctes)
            columns = tuple(f"column{position}" for position in range(1, width + 1))
        else:
            columns = self._project(stmt, scopes, ctes, redact)

        for clause in (
            stmt.whereClause, stmt.havingClause, stmt.windowClause, stmt.limitOffset, stmt.limitCount,
        ):
            self._expression(clause, scopes, ctes)

        # Set operations expose their output columns to ORDER BY.
        sort_scopes = () if stmt.op != SetOperation.SETOP_NONE else scopes
        for sort in stmt.sortClause or ():
            self._sort_or_group(sort.node, columns, sort_scopes, ctes)
        for item in stmt.distinctClause or ():
            self._sort_or_group(item, columns, scopes, ctes)
        for item in stmt.groupClause or ():
            self._sort_or_group(item, columns, scopes, ctes, grouping=True)
        return columns

    def _from_item(
        self, item: ast.Node, ctes: _CTEs, outer: _Scopes, preceding: _Scope
    ) -> _Scope:
        if isinstance(item, ast.RangeVar):
            return (self._relation(item, ctes),)

        if isinstance(item, ast.RangeSubselect):
            columns = self.select(item.subquery, ctes, (preceding, *outer) if item.lateral else outer)
            if item.alias is None:
                raise RestrictedColumnError("A FROM subquery requires an alias")
            return (_Source((item.alias.aliasname,), _rename_columns(columns, item.alias.colnames)),)

        if isinstance(item, ast.JoinExpr):
            if item.isNatural or item.usingClause or item.alias:
                raise RestrictedColumnError("Use an explicit JOIN ON with named input relations")
            left = self._from_item(item.larg, ctes, outer, preceding)
            right = self._from_item(item.rarg, ctes, outer, preceding + left)
            joined = left + right
            self._expression(item.quals, (joined, *outer), ctes)
            return joined

        if isinstance(item, ast.RangeTableSample):
            source = self._relation(item.relation, ctes)
            method = tuple(part.sval for part in item.method)
            if source.table is None or method not in {
                ("system",), ("bernoulli",), ("pg_catalog", "system"), ("pg_catalog", "bernoulli"),
            }:
                raise RestrictedColumnError("TABLESAMPLE requires a base relation and a built-in method")
            self._expression((item.args, item.repeatable), outer, ctes)
            # The execution search path starts with pg_catalog. RawStream
            # cannot serialize qualified TABLESAMPLE method names.
            item.method = _identifiers((method[-1],))
            return (source,)

        if isinstance(item, ast.RangeFunction):
            return (self._range_function(item, (preceding, *outer), ctes),)

        raise RestrictedColumnError(f"Cannot resolve FROM source {type(item).__name__}")

    def _relation(self, item: ast.RangeVar, ctes: _CTEs) -> _Source:
        if item.catalogname:
            raise RestrictedColumnError("Database-qualified relations are not supported")
        name = item.alias.aliasname if item.alias else item.relname
        aliases = item.alias.colnames if item.alias else None
        if item.schemaname is None and item.relname in ctes:
            return _Source((name,), _rename_columns(ctes[item.relname], aliases))

        schema = item.schemaname or "public"
        if schema.lower() in BLOCKED_SCHEMAS or schema.lower().startswith(("pg_temp_", "pg_toast_")):
            raise SystemCatalogError(f"Access to schema {schema} is blocked")
        if item.relname.lower().startswith(BLOCKED_TABLE_PREFIXES):
            raise SystemCatalogError(f"Access to table {item.relname} is blocked")

        columns = self.allowlist.database_table_columns(schema, item.relname)
        if columns is None:
            raise RestrictedColumnError(f"Table {schema}.{item.relname} is not in the allowlist")
        if aliases:
            raise RestrictedColumnError("Column alias lists on base relations cannot be resolved")

        item.schemaname = schema
        table = item.relname if schema == "public" else f"{schema}.{item.relname}"
        self.result.tables_accessed.append(table)
        qualifier = (name,) if item.alias else (schema, name)
        return _Source(qualifier, columns, table)

    def _range_function(self, item: ast.RangeFunction, scopes: _Scopes, ctes: _CTEs) -> _Source:
        self._expression(item.functions, scopes, ctes)
        if item.ordinality or item.is_rowsfrom or item.coldeflist or len(item.functions) != 1:
            raise RestrictedColumnError("Cannot resolve columns returned by FROM function")
        call, definitions = item.functions[0]
        if (
            definitions or not isinstance(call, ast.FuncCall)
            or tuple(part.sval for part in call.funcname) != ("generate_series",)
        ):
            raise RestrictedColumnError("Cannot resolve columns returned by FROM function")

        # This built-in has one scalar output whose name is determined by the alias.
        call.funcname = _identifiers(("pg_catalog", "generate_series"))
        name = item.alias.aliasname if item.alias else "generate_series"
        aliases = item.alias.colnames if item.alias else None
        return _Source((name,), _rename_columns((name,), aliases))

    def _project(self, stmt: ast.SelectStmt, scopes: _Scopes, ctes: _CTEs, redact: bool) -> _Output:
        targets = []
        for target in stmt.targetList or ():
            value = target.val
            if isinstance(value, ast.ColumnRef) and isinstance(value.fields[-1], ast.A_Star):
                targets.extend(self._expand_star(value, scopes, redact))
                self.result.was_rewritten = True
                continue

            hidden = [] if redact else None
            self._expression(value, scopes, ctes, hidden)
            if hidden:
                self.result.columns_redacted.extend(hidden)
                name = target.name or (
                    value.fields[-1].sval if isinstance(value, ast.ColumnRef) else "redacted"
                )
                target = _redacted_target(name)
                self.result.was_rewritten = True
            targets.append(target)

        if stmt.targetList is not None:
            stmt.targetList = tuple(targets)
        return tuple(_output_name(target) for target in targets)

    def _expand_star(self, ref: ast.ColumnRef, scopes: _Scopes, markers: bool) -> list[ast.ResTarget]:
        qualifier = tuple(field.sval for field in ref.fields[:-1])
        sources = (_qualified_source(qualifier, scopes),) if qualifier else scopes[0]
        if not sources:
            raise RestrictedColumnError("Wildcard has no visible source")

        targets = []
        for source in sources:
            _qualified_source(source.qualifier, scopes)
            if None in source.columns or len(set(source.columns)) != len(source.columns):
                raise RestrictedColumnError("Wildcard output requires unique, known column names")
            for column in source.columns:
                if source.table is not None:
                    self.result.columns_accessed.append(f"{source.table}.{column}")
                targets.append(ast.ResTarget(
                    val=ast.ColumnRef(fields=_identifiers((*source.qualifier, column))),
                ))
            if markers and source.table is not None:
                name = f"_{source.name}_redacted_columns"
                targets.append(_redacted_target(name.encode()[:63].decode(errors="ignore")))
        if not targets:
            raise RestrictedColumnError("Wildcard has no approved columns")
        return targets

    def _expression(
        self,
        node: object,
        scopes: _Scopes,
        ctes: _CTEs,
        hidden: list[str] | None = None,
    ) -> None:
        if isinstance(node, ast.ColumnRef):
            source, column = _resolve_column(node, scopes)
            if source.table is not None:
                name = f"{source.table}.{column}"
                if column in source.columns:
                    self.result.columns_accessed.append(name)
                elif hidden is not None:
                    hidden.append(name)
                else:
                    raise RestrictedColumnError(f"Column {name} is restricted in this expression")
            return

        if isinstance(node, ast.SubLink):
            self._expression(node.testexpr, scopes, ctes)
            self.select(node.subselect, ctes, scopes)
            return

        if isinstance(node, ast.SelectStmt):
            self.select(node, ctes, scopes)
            return

        if isinstance(node, ast.XmlExpr):
            raise DisallowedFunctionError("XML functions are not allowed")

        if isinstance(node, ast.A_Indirection) and any(
            not isinstance(index, ast.A_Indices) for index in node.indirection
        ):
            raise RestrictedColumnError("Record field references cannot be resolved")

        if isinstance(node, ast.FuncCall):
            _validate_function(node, self.allowlist)

        if isinstance(node, tuple):
            for child in node:
                self._expression(child, scopes, ctes, hidden)
        elif isinstance(node, ast.Node):
            for attr in node.__slots__:
                # These expressions affect which rows an aggregate or window uses.
                child_hidden = None if isinstance(node, ast.FuncCall) and attr in {
                    "agg_filter", "agg_order", "over",
                } else hidden
                self._expression(getattr(node, attr), scopes, ctes, child_hidden)

    def _sort_or_group(
        self, node: ast.Node, columns: _Output, scopes: _Scopes, ctes: _CTEs, *, grouping: bool = False
    ) -> None:
        if isinstance(node, ast.GroupingSet):
            for item in node.content or ():
                self._sort_or_group(item, columns, scopes, ctes, grouping=grouping)
            return
        if isinstance(node, ast.A_Const) and isinstance(node.val, ast.Integer):
            if not 1 <= node.val.ival <= len(columns):
                raise RestrictedColumnError("Projection position is out of range")
            return
        if isinstance(node, ast.ColumnRef) and len(node.fields) == 1:
            field = node.fields[0]
            # GROUP BY prefers input names. ORDER BY and DISTINCT ON prefer output names.
            if isinstance(field, ast.String) and not (
                grouping and any(source.may_have_column(field.sval) for source in scopes[0])
            ):
                count = columns.count(field.sval)
                if None in columns or count > 1:
                    raise RestrictedColumnError("Use unique, explicit projection aliases or a qualified column")
                if count == 1:
                    return
        self._expression(node, scopes, ctes)


def _qualified_source(qualifier: tuple[str, ...], scopes: _Scopes) -> _Source:
    if len(qualifier) not in (1, 2):
        raise RestrictedColumnError("Unsupported column qualifier")
    for scope in scopes:
        matches = [s for s in scope if qualifier == (s.name,) or qualifier == s.qualifier]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise RestrictedColumnError("Ambiguous column qualifier")
    raise RestrictedColumnError("Unknown column qualifier")


def _resolve_column(ref: ast.ColumnRef, scopes: _Scopes) -> tuple[_Source, str]:
    if not ref.fields or any(not isinstance(field, ast.String) for field in ref.fields):
        raise RestrictedColumnError("Whole-row references and expression wildcards are not allowed")
    *qualifier, column = (field.sval for field in ref.fields)
    if qualifier:
        source = _qualified_source(tuple(qualifier), scopes)
    else:
        source = _unqualified_source(column, scopes)
    if source.table is None and (None in source.columns or source.columns.count(column) != 1):
        raise RestrictedColumnError("Column has no unique output in the referenced source")
    return source, column


def _unqualified_source(column: str, scopes: _Scopes) -> _Source:
    for scope in scopes:
        matches = [source for source in scope if source.may_have_column(column)]
        if len(matches) > 1:
            raise RestrictedColumnError("Qualify columns that could belong to multiple sources")
        if matches:
            source = matches[0]
            if column not in source.columns and any(s.name == column for s in scope):
                raise RestrictedColumnError("Whole-row references are not allowed")
            return source
    raise RestrictedColumnError("Column has no visible source")


def _rename_columns(columns: _Output, aliases: tuple | None) -> _Output:
    if not aliases:
        return columns
    if len(aliases) > len(columns):
        raise RestrictedColumnError("Source has fewer columns than its alias list")
    return tuple(alias.sval for alias in aliases) + columns[len(aliases):]


def _output_name(target: ast.ResTarget) -> str | None:
    if target.name is not None:
        return target.name
    value = target.val
    if isinstance(value, ast.ColumnRef):
        return value.fields[-1].sval
    if isinstance(value, ast.FuncCall):
        return value.funcname[-1].sval
    if isinstance(value, ast.A_Expr):
        return "nullif" if value.kind == A_Expr_Kind.AEXPR_NULLIF else "?column?"
    if isinstance(value, ast.A_Const):
        return "?column?"
    return None


def _identifiers(parts: tuple[str, ...]) -> tuple[ast.String, ...]:
    return tuple(ast.String(sval=part) for part in parts)


def _redacted_target(name: str) -> ast.ResTarget:
    return ast.ResTarget(name=name, val=ast.A_Const(val=ast.String(sval="[REDACTED]")))


def _validate_function(func_call: ast.FuncCall, allowlist: Allowlist) -> None:
    parts = tuple(part.sval for part in func_call.funcname)
    name = parts[-1].upper()
    if name in ALWAYS_BLOCKED_FUNCTIONS:
        raise DisallowedFunctionError(f"Function {name} is not allowed")
    if len(parts) > 1 and parts[0].lower() in BLOCKED_SCHEMAS:
        raise SystemCatalogError(f"Function in schema {parts[0]} is not allowed")
    if not allowlist.is_function_allowed(name):
        raise DisallowedFunctionError(f"Function {name} is not on the allowed list")
    if func_call.agg_star and name != "COUNT":
        raise DisallowedFunctionError(f"Star (*) in function arguments is not allowed: {name}(*)")
