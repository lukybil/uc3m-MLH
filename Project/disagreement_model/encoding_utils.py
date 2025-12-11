"""
Utility module to handle Windows UTF-8 encoding for stdout/stderr.
This should be imported once at the entry point of the application.
"""

import sys

# Flag to prevent double-wrapping
_ENCODING_FIXED = False


def fix_windows_encoding():
    """
    Fix Windows console encoding to UTF-8.
    Safe to call multiple times - only wraps streams once.
    """
    global _ENCODING_FIXED

    if _ENCODING_FIXED:
        return  # Already fixed, don't wrap again

    if sys.platform == "win32":
        import io

        # Only wrap if not already wrapped
        if (
            not isinstance(sys.stdout, io.TextIOWrapper)
            or sys.stdout.encoding != "utf-8"
        ):
            try:
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
            except (AttributeError, ValueError):
                pass  # Already wrapped or not available

        if (
            not isinstance(sys.stderr, io.TextIOWrapper)
            or sys.stderr.encoding != "utf-8"
        ):
            try:
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer, encoding="utf-8", errors="replace"
                )
            except (AttributeError, ValueError):
                pass  # Already wrapped or not available

        _ENCODING_FIXED = True
