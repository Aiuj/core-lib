"""Multilingual date formatting utilities for Excel cells.

Provides two public classes built on Babel's CLDR data:

* :class:`ExcelDateFormatter` — converts Excel ``number_format`` codes to
  locale-aware date strings.  Replaces the previous English-only strftime
  approach in ``ExcelManager``.

* :class:`MonthExpander` — detects abbreviated month-year patterns (e.g.
  "Jun-23", "janv.-23") in text and expands them to include the full month
  name across all configured locales.  Used to improve cosine similarity when
  embedding financial table cells.

Both classes delegate to Babel (``babel.dates`` / ``babel.Locale``) so any of
the ~500 CLDR locales is supported out of the box, with no hand-crafted month
dictionaries.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Sequence, Union

from babel import Locale
from babel.dates import format_date as _babel_format_date, get_month_names as _get_month_names


class ExcelDateFormatter:
    """Format Excel cell date values using their ``number_format`` codes.

    Converts Excel format codes such as ``mmm-yy``, ``mmmm-yyyy``, or ``d-mmm``
    to locale-aware formatted strings via Babel's CLDR data.  Falls back to
    ISO 8601 (``YYYY-MM-DD``) for unrecognised codes.

    This is a drop-in, locale-aware replacement for the previous English-only
    ``_format_excel_date`` static helper in ``ExcelManager``.

    Example::

        fmt = ExcelDateFormatter(locale='fr')
        fmt.format(date(2023, 6, 1), 'mmm-yy')    # → 'juin-23'
        fmt.format(date(2023, 6, 1), 'mmmm-yyyy') # → 'juin 2023'

        fmt_en = ExcelDateFormatter()              # default: 'en'
        fmt_en.format(date(2023, 6, 1), 'mmm-yy') # → 'Jun-23'
    """

    # Excel number_format cleanup patterns (applied in order, case-insensitive)
    _CLEANUP_PATTERNS: tuple[re.Pattern, ...] = (
        re.compile(r'\[\$[^\]]*\]'),           # [$-en-US] or [$409]
        re.compile(r'\[[a-z]+\]', re.I),        # [Red], [Green], …
        re.compile(r'\[[<>=!][^\]]+\]'),         # [>=0], [<0]
    )
    _MULTI_DASH = re.compile(r'[-\s]{2,}')

    # Maps (normalised Excel format code pattern) → Babel CLDR format string.
    # Excel uses lowercase letters (mmm, mmmm, d, yy, yyyy);
    # Babel uses uppercase M/d/y (MMM, MMMM, d, yy, yyyy).
    _FORMAT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
        # Month-Year
        (re.compile(r'^mmm[-/]yy$'),     'MMM-yy'),
        (re.compile(r'^mmm[-/]yyyy$'),   'MMM-yyyy'),
        (re.compile(r'^mmmm[-/]yy$'),    'MMMM-yy'),
        (re.compile(r'^mmmm[-/]yyyy$'),  'MMMM-yyyy'),
        # Month-Day  (no-leading-zero day via 'd' in Babel CLDR)
        (re.compile(r'^mmm[-/]d+$'),     'MMM-d'),
        (re.compile(r'^mmmm[-/]d+$'),    'MMMM-d'),
        (re.compile(r'^d+[-/]mmm$'),     'd-MMM'),
        (re.compile(r'^d+[-/]mmmm$'),    'd-MMMM'),
        # Year-Month
        (re.compile(r'^yyyy[-/]mm?$'),   'yyyy-MM'),
        # Full date with abbreviated month name
        (re.compile(r'^d+[-/]mmm[-/]yy$'),    'd-MMM-yy'),
        (re.compile(r'^d+[-/]mmm[-/]yyyy$'),  'd-MMM-yyyy'),
    )

    def __init__(self, locale: str = 'en') -> None:
        """
        Args:
            locale: IETF BCP 47 locale tag (e.g. ``'en'``, ``'fr'``, ``'de'``,
                    ``'fr_CA'``).  Defaults to English.
        """
        self._locale = Locale.parse(locale)

    def _clean_format(self, number_format: str) -> str:
        fmt = (number_format or '').lower().strip()
        for pat in self._CLEANUP_PATTERNS:
            fmt = pat.sub('', fmt)
        fmt = self._MULTI_DASH.sub('-', fmt).strip('-').strip()
        return fmt

    def format(self, value: Union[date, datetime], number_format: str = '') -> str:
        """Format *value* using the Excel *number_format* code.

        Args:
            value: A :class:`~datetime.date` or :class:`~datetime.datetime`
                instance (the already-parsed cell value from openpyxl).
            number_format: The cell's ``number_format`` string from openpyxl
                (e.g. ``'mmm-yy'``, ``'[$-en-US]mmmm d\\, yyyy'``).

        Returns:
            A formatted date string, or ISO 8601 for unrecognised formats.
        """
        fmt = self._clean_format(number_format)
        d = value.date() if isinstance(value, datetime) else value

        for pattern, babel_fmt in self._FORMAT_PATTERNS:
            if pattern.match(fmt):
                try:
                    return _babel_format_date(d, format=babel_fmt, locale=self._locale)
                except Exception:
                    break  # fall through to ISO fallback

        # ISO fallback
        if isinstance(value, datetime) and (value.hour or value.minute or value.second):
            return value.strftime('%Y-%m-%d %H:%M')
        return value.strftime('%Y-%m-%d')


class MonthExpander:
    """Expand abbreviated month-year patterns to full names across multiple locales.

    Scans text for patterns like ``"Jun-23"`` (English) or ``"janv.-23"``
    (French) and appends the full month name and four-digit year in all
    configured locales, e.g.::

        "Jun-23"    → "Jun-23 (June 2023 / juin 2023)"
        "janv.-23"  → "janv.-23 (January 2023 / janvier 2023)"
        "No dates"  → "No dates"  (unchanged)

    This is used to pre-process embedding text so that natural-language queries
    ("What is the June 2023 EBITDA?") match abbreviated cell values ("Jun-23")
    with high cosine similarity.

    Uses Babel's CLDR data — any locale supported by Babel (700+) works.

    Args:
        locales: Ordered list of IETF BCP 47 locale tags.  Abbreviated month
                 forms from *all* locales are recognised; full names from *all*
                 locales appear in the expanded annotation.  Defaults to
                 ``('en', 'fr')``.
    """

    def __init__(self, locales: Sequence[str] = ('en', 'fr')) -> None:
        self._locales = list(locales)
        # normalised_abbr → month number (1-12)
        self._abbr_to_num: dict[str, int] = {}
        # locale_str → {month_number → full (wide) name}
        self._num_to_wide: dict[str, dict[int, str]] = {}
        self._pattern: re.Pattern | None = None
        self._build()

    def _build(self) -> None:
        all_abbr: set[str] = set()

        for lang in self._locales:
            loc = Locale.parse(lang)

            # Abbreviated month names (key = 1-based month number)
            for month_num, abbr in _get_month_names(width='abbreviated', locale=loc).items():
                # Strip trailing dot that some locales add (French: 'janv.', 'févr.')
                normalised = abbr.rstrip('.').lower()
                # First locale wins for a given abbreviation form
                self._abbr_to_num.setdefault(normalised, month_num)
                all_abbr.add(normalised)

            # Wide (full) month names
            self._num_to_wide[lang] = {
                month_num: name
                for month_num, name in _get_month_names(width='wide', locale=loc).items()
            }

        if not all_abbr:
            return

        # Build regex: longest alternatives first so "janv" is tried before "jan"
        abbr_alts = '|'.join(
            re.escape(a) for a in sorted(all_abbr, key=len, reverse=True)
        )
        # \.? handles an optional trailing dot (French: "janv." before the separator)
        # (?<![a-zA-Z]) prevents matching inside words
        self._pattern = re.compile(
            rf'(?<![a-zA-Z])({abbr_alts})\.?[-/](\d{{2,4}})\b',
            re.IGNORECASE,
        )

    def expand(self, value: str) -> str:
        """Expand month-year abbreviations in *value* to include full names.

        Args:
            value: A cell value string (e.g. a column value from an Excel table).

        Returns:
            The value with month-year patterns annotated, or the original string
            if no recognised patterns are found.
        """
        if self._pattern is None:
            return value

        def _replace(m: re.Match) -> str:
            raw_abbr = m.group(1)
            year_part = m.group(2)
            year = f'20{year_part}' if len(year_part) == 2 else year_part
            month_num = self._abbr_to_num.get(raw_abbr.rstrip('.').lower())
            if month_num is None:
                return m.group(0)

            # Collect full names from each locale, deduplicating identical strings
            full_names: list[str] = []
            seen: set[str] = set()
            for lang in self._locales:
                wide = self._num_to_wide.get(lang, {}).get(month_num)
                if wide and wide not in seen:
                    full_names.append(f'{wide} {year}')
                    seen.add(wide)

            if not full_names:
                return m.group(0)

            return f'{m.group(0)} ({" / ".join(full_names)})'

        return self._pattern.sub(_replace, value)
