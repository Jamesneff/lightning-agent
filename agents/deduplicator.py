"""
deduplicator.py — Filters out startup profiles already seen or internally duplicated.

Responsibilities:
  - Accept structured startup profiles from parser.py before scoring.
  - Maintain (or query) a persistent seen-set: a local cache file or a
    lightweight DB (e.g. SQLite / Redis) keyed on canonical startup URL.
  - Detect near-duplicates within the current batch using fuzzy name/URL
    matching to handle slight variations across sources.
  - Mark profiles as 'duplicate' or drop them from the pipeline entirely.
  - Update the seen-set with newly passing profiles so future runs skip them.

Primary interface:
    deduplicate(profiles: list[dict]) -> list[dict]
        Returns only profiles not previously seen and not internally duplicated
        within the current batch.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_SEEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seen_companies.json")


def load_seen() -> set[str]:
    """Load the set of already-seen company names from disk.

    Returns a set of lowercased company name strings.
    Returns an empty set if the file does not exist yet.
    """
    path = os.path.normpath(_SEEN_PATH)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("seen_companies.json is not a list — resetting to empty")
            return set()
        return {name.lower() for name in data if isinstance(name, str)}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read seen_companies.json: %s — starting fresh", exc)
        return set()


def save_seen(seen: set[str]) -> None:
    """Persist the seen set to disk as a sorted JSON list of lowercased names.

    Creates the data/ directory if it does not exist.
    """
    path = os.path.normpath(_SEEN_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(seen), f, indent=2)
        logger.debug("Saved %d seen companies to %s", len(seen), path)
    except OSError as exc:
        logger.error("Could not write seen_companies.json: %s", exc)


def deduplicate(profiles: list[dict]) -> list[dict]:
    """Filter profiles to only those whose company has not been seen before.

    Deduplicates both against the persistent seen file and within the current
    batch (so the same company appearing in two articles is included once).

    Args:
        profiles: List of profile dicts, each with at minimum a 'company_name' key.

    Returns:
        Subset of profiles that are new, preserving original order.
        The caller is responsible for calling save_seen() after processing the
        returned profiles.
    """
    seen = load_seen()
    new_profiles: list[dict] = []
    batch_seen: set[str] = set()

    for profile in profiles:
        name = (profile.get("company_name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen or key in batch_seen:
            logger.debug("Skipping duplicate: %s", name)
            continue
        batch_seen.add(key)
        new_profiles.append(profile)

    logger.info(
        "Deduplicated %d → %d profiles (%d already seen, %d intra-batch dupes removed)",
        len(profiles),
        len(new_profiles),
        sum(1 for p in profiles if (p.get("company_name") or "").strip().lower() in seen),
        len(profiles) - len(new_profiles) - sum(
            1 for p in profiles if (p.get("company_name") or "").strip().lower() in seen
        ),
    )
    return new_profiles
