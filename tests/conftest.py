"""Test configuration: Hypothesis profiles.

``ci`` (selected automatically when the ``CI`` environment variable is set) is
derandomized, so every CI run checks the same examples. ``dev`` explores new random
examples on every local run, and ``thorough`` runs many more. Select a profile with
``HYPOTHESIS_PROFILE=<name>``.
"""

import os

from hypothesis import settings

settings.register_profile(
    "ci", max_examples=40, deadline=None, derandomize=True, database=None, print_blob=True
)
settings.register_profile("dev", max_examples=40, deadline=None)
settings.register_profile("thorough", max_examples=1000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci" if os.environ.get("CI") else "dev"))
