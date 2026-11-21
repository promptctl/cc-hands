"""A bearer token as hands' servers check one: the header that carries it gets in, and nothing else does."""

import pytest

from hands.core.bearer import bearer, bears


def test_the_header_that_carries_the_token_gets_in() -> None:
    assert bears(bearer("the-token"), "the-token")


@pytest.mark.parametrize("authorization", ["the-token", "bearer the-token", "Bearer  the-token", "Bearer the-token ", "Bearer other", "Bearer ", ""])
def test_any_other_header_is_refused(authorization: str) -> None:
    assert not bears(authorization, "the-token")


def test_a_header_whose_bytes_are_not_utf8_is_refused_rather_than_raising() -> None:
    # As aiohttp hands over a header byte that is no UTF-8.
    assert not bears(b"Bearer \xff".decode("utf-8", "surrogateescape"), "the-token")
