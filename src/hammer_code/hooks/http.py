"""Anonymous, bounded standard-library HTTP actions."""

from __future__ import annotations

import asyncio
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


async def request(
    *, url: str, method: str, body: str, headers: dict[str, str], timeout: int
) -> tuple[bool, str]:
    return await asyncio.to_thread(
        _request, url=url, method=method, body=body, headers=headers, timeout=timeout
    )


def _request(
    *, url: str, method: str, body: str, headers: dict[str, str], timeout: int
) -> tuple[bool, str]:
    try:
        data = body.encode("utf-8") if body else None
        response_request = Request(url, data=data, headers=headers, method=method)
        with urlopen(response_request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(2048)
            charset = response.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")[:500]
            return True, text
    except HTTPError as exc:
        return False, f"HTTP status {exc.code}"
    except (TimeoutError, URLError, OSError, UnicodeError):
        return False, "HTTP request failed"
