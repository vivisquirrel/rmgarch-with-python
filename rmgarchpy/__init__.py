"""
rmgarchpy - Python port of the R package rmgarch.

Implements:
  - DCC-GARCH (fit, filter, simulate, forecast; gaussian & semiparametric likelihood)
  - Dynamic Copula cGARCH (mvnormal / mvt, static & dynamic)
  - mdcopula toolbox (Kendall tau, tail dependence, copula families, GoF)
  - GO-GARCH
  - FastICA / Radical
  - VARX
  - Diagnostic tests (DCCtest, MardiaTest, rshape, rskew)

Usage:
    import rmgarchpy as rmg
    fit = rmg.dccfit(data, p=1, q=1)
"""

from . import utils
from .ugarch import ugarchfit, ugarchfilter, ugarchsim
from .dcc import dccfit, dccfilter, dccsim, dccforecast, dccuncor
from .cgarch import cgarchfit, cgarchfilter, cgarchsim
from .mdcopula import (
    mdKendall, mdTailDep, mdCopulaParam, mdCopulaGof, mdCopulaSim,
    mdStaticCopula, mdDynamicCopula,
)
from .go_garch import go_garch_fit, go_garch_filter
from .ica import fast_ica, radical
from .varx import varx
from .diagnostics import dcc_test, mardia_test, rshape, rskew

__version__ = "0.1.0"

__all__ = [
    "ugarchfit", "ugarchfilter", "ugarchsim",
    "dccfit", "dccfilter", "dccsim", "dccforecast", "dccuncor",
    "cgarchfit", "cgarchfilter", "cgarchsim",
    "mdKendall", "mdTailDep", "mdCopulaParam", "mdCopulaGof", "mdCopulaSim",
    "mdStaticCopula", "mdDynamicCopula",
    "go_garch_fit", "go_garch_filter",
    "fast_ica", "radical",
    "varx",
    "dcc_test", "mardia_test", "rshape", "rskew",
    "utils",
]