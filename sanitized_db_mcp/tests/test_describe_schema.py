"""Tests for the describe_schema tool.

An agent cannot discover what is queryable any other way: information_schema is
blocked, and a nonexistent table returns the same error as a restricted one.
"""

from sanitized_db_mcp.allowlist import Allowlist


class TestTableListing:
    def test__no_pattern_lists_every_queryable_table(self, sample_allowlist):
        out = sample_allowlist.describe()

        assert "accounts_profile" in out
        assert "ecomm_order" in out
        assert "auth_user" in out
        assert "3 queryable tables" in out

    def test__listing_is_sorted_for_stable_output(self, sample_allowlist):
        out = sample_allowlist.describe()
        names = [line for line in out.splitlines() if not line.startswith("/*")]
        assert names == sorted(names)

    def test__listing_points_at_the_next_call(self, sample_allowlist):
        # The agent needs to know columns come from a second call, not guesswork.
        assert "table name for its columns" in sample_allowlist.describe()


class TestColumnListing:
    def test__exact_table_name_returns_its_columns(self, sample_allowlist):
        out = sample_allowlist.describe("ecomm_order")

        assert "status" in out
        assert "order_number" in out
        assert "create_date" in out

    def test__hidden_columns_are_absent(self, sample_allowlist):
        # email and payload are not visible on ecomm_order in the fixture. An agent
        # that saw them listed would write SQL the sanitizer then rejects.
        out = sample_allowlist.describe("ecomm_order")

        assert "email" not in out
        assert "payload" not in out

    def test__placeholders_are_shown(self, sample_allowlist):
        out = sample_allowlist.describe("ecomm_order")
        assert "'[REDACTED]'" in out or "0" in out

    def test__says_that_absent_columns_cannot_be_referenced(self, sample_allowlist):
        assert "cannot be referenced" in sample_allowlist.describe("ecomm_order")


class TestPatternMatching:
    def test__wildcard_narrows_the_table_list(self, sample_allowlist):
        out = sample_allowlist.describe("accounts_*")

        assert "accounts_profile" in out
        assert "ecomm_order" not in out

    def test__unique_prefix_without_wildcard_resolves_to_the_table(self, sample_allowlist):
        out = sample_allowlist.describe("ecomm_")

        # Only one ecomm_ table in the fixture, so this should give columns.
        assert "order_number" in out

    def test__ambiguous_prefix_lists_candidates_instead_of_guessing(self, sample_allowlist):
        out = sample_allowlist.describe("a")

        assert "accounts_profile" in out
        assert "auth_user" in out

    def test__no_match_is_an_error_that_says_what_to_do_next(self, sample_allowlist):
        out = sample_allowlist.describe("prescriptions_prescriptionfill")

        assert out.startswith("Error:")
        assert "no arguments for the full list" in out

    def test__bare_noun_guess_fails_clearly(self, sample_allowlist):
        # The real failure mode: an agent guesses "orders" rather than "ecomm_order".
        out = sample_allowlist.describe("orders")
        assert out.startswith("Error:")
