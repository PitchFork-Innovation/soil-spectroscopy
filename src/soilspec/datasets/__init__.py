"""Curated AOI/window registries and data-build drivers.

Submodules:

- :mod:`soilspec.datasets.initial_aois` — registry of AOI+window pairs
  for the first joint-training cut. Coordinates are loaded lazily from
  ``data/registry/initial_aois.json``, which is produced by the
  regenerator below.
- :mod:`soilspec.datasets.regenerate_aois` — derive AOIs from the
  ground-truth CSVs (so each AOI is guaranteed to contain a real
  measurement). Run once after the GT CSVs land.
- :mod:`soilspec.datasets.ismn_to_csv` — convert ISMN archives to the
  flat CSV ``ISMNAdapter`` reads.
- :mod:`soilspec.datasets.lucas_to_csv` — convert ESDAC LUCAS 2018 to
  the flat CSV ``LUCASAdapter`` reads.
- :mod:`soilspec.datasets.build` — driver that walks the registry,
  fetches S1/S2 from Planetary Computer, joins with ISMN/LUCAS labels,
  and materializes a :class:`RasterTrainingExamples` to disk.

The eager re-exports below are limited to classes/constants that don't
trigger a registry load. The AOI tuples themselves
(``ISMN_AOIS``/``LUCAS_AOIS``/``INITIAL_AOIS``) live on the
``initial_aois`` submodule and resolve on first attribute access.
"""

from .initial_aois import (
    AOIWindow,
    ISMN_NH_WINDOW,
    ISMN_SH_WINDOW,
    LUCAS_WINDOW,
    RegistryNotGeneratedError,
)

__all__ = [
    "AOIWindow",
    "ISMN_NH_WINDOW",
    "ISMN_SH_WINDOW",
    "LUCAS_WINDOW",
    "RegistryNotGeneratedError",
]
