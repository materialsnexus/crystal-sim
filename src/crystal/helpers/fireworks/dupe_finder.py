# noqa: D100
from dataclasses import dataclass
from typing import Any

from fireworks.features.dupefinder import DupeFinderBase

DEDUPE_KEY = "dedupe_key"


@dataclass(kw_only=True)
class DupeFinderBySpecKey(DupeFinderBase):  # noqa: D101
    _fw_name: str = "DupeFinderBySpecKey"
    spec_key: str = DEDUPE_KEY

    def query(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Mongo query for shortlisting possible duplicates."""
        # If spec key does not exist in the spec, the query should not find anything in the database
        if self.spec_key not in spec:
            return {"_id": None}
        return {
            "$and": [
                {"launches": {"$ne": []}},
                {f"spec.{self.spec_key}": spec[self.spec_key]},
            ]
        }

    def verify(self, spec1: dict[str, Any], spec2: dict[str, Any]) -> bool:  # noqa: D102
        spec1_value = spec1.get(self.spec_key)
        spec2_value = spec2.get(self.spec_key)
        if spec1_value is None or spec2_value is None:
            return False
        return spec1.get(self.spec_key) == spec2.get(self.spec_key)
