"""
Makes console output survive a non-Unicode stdout.

WHY THIS EXISTS
---------------
These scripts print real KArSL labels, which are Arabic. On Windows a Python
process that is not attached to a UTF-8 console gets a legacy code page for
stdout -- cp1252 under the default Windows Terminal locale, which is what
`python -m pytest` inherits. Printing an Arabic label then raises:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u0628'

That is not a cosmetic failure. `import_karsl.py` prints each sign's label as
it imports it and writes the output JSON only after the loop finishes, so the
exception kills the process BEFORE the registry file is written, and every
later assertion fails with a FileNotFoundError pointing at a missing output
file rather than at the real cause. Thirteen importer tests failed this way on
Windows while the same code passed on a UTF-8 machine.

WHAT IT DOES
------------
Leaves the stream's ENCODING alone and only sets its error handler to
"replace". Two reasons for that choice over forcing UTF-8:

  * On a console that already handles UTF-8, nothing changes -- Arabic still
    prints as Arabic.
  * On a legacy code page, an unencodable character becomes "?" instead of
    either crashing or emitting UTF-8 bytes that a cp1252 console would render
    as mojibake. "?" is honest about what the console cannot show.

WHAT IT DOES NOT DO
-------------------
It does not touch file I/O. Every file these scripts write is opened with an
explicit `encoding="utf-8"`, so the JSON registry keeps the real Arabic labels
byte for byte. This only affects what goes to a terminal.
"""
from __future__ import annotations

import io
import sys


def _already_unicode(stream) -> bool:
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    return encoding in ("utf8", "utf8sig", "utf16", "utf32")


def make_stdio_unicode_safe() -> None:
    """Ensures printing a non-ASCII character can never raise.

    Safe to call more than once, and safe when stdout has been replaced by a
    test harness or redirected to something that is not a TextIOWrapper -- in
    that case it does nothing rather than risking the caller's stream.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or _already_unicode(stream):
            continue

        # Python 3.7+: change the error handler in place, keeping the console's
        # own encoding and everything else about the stream.
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
                continue
            except (ValueError, OSError, AttributeError):
                pass

        # Fallback for a stream without reconfigure() but with a real byte
        # buffer underneath. Anything else is left untouched.
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        try:
            setattr(
                sys,
                name,
                io.TextIOWrapper(
                    buffer,
                    encoding=getattr(stream, "encoding", None) or "utf-8",
                    errors="replace",
                    line_buffering=True,
                ),
            )
        except (ValueError, OSError, AttributeError):
            pass
