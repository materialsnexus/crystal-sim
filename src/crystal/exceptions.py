# noqa: D100


class OverlappingAtomsException(Exception):
    """Raised when generated interfaces contain overlapping atoms."""


class NoInterfaceMatchException(Exception):
    """Raised when no ZSL match is found for an interface slab pair."""


class UnsupportedTerminationLayerException(Exception):
    """Raised when a requested slab termination layer is not available."""
