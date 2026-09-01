"""All source adapters import and expose the expected surface.

Enrichment adapters (M1b) are skeletons — we only assert they load and that the
documented entrypoint raises NotImplementedError with a hint, so ``run_ingest``
plumbing stays honest.
"""

import pytest

from ingest.sources.fbref import FbrefAdvancedStats
from ingest.sources.rfs import RfsYouthCallups
from ingest.sources.sofascore import SofascoreRatings
from ingest.sources.youfl import Youfl


def test_youfl_is_deferred():
    with pytest.raises(NotImplementedError, match="M1b"):
        Youfl().fetch_player("x")


@pytest.mark.parametrize(
    "call",
    [
        lambda: RfsYouthCallups().iter_callups("U-17", 2022),
        lambda: SofascoreRatings().player_season_ratings(1),
        lambda: FbrefAdvancedStats().player_stats("abc"),
    ],
)
def test_enrichment_adapters_declare_todo(call):
    with pytest.raises(NotImplementedError, match="M1b"):
        call()
