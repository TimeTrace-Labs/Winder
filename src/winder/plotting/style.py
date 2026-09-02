"""One shared style for every publication figure: `scienceplots`' `science` + `no-latex`
styles, plus editable (not Type-3 bitmap) PDF text.

Verified on this machine (`scienceplots==2.1.1`, `matplotlib==3.11.1`) by reading the installed
`.mplstyle` files directly, not assumed: `science.mplstyle` alone sets `text.usetex: True` (it
would fail hard on a machine with no system LaTeX install), serif fonts, 0.5pt axes, inward
ticks, and a frameless legend. `no-latex.mplstyle`, applied second so it overrides, sets
`text.usetex: False` and `font.serif: STIXGeneral` (bundled with matplotlib itself -- no
OS-level font hunting, no system-LaTeX dependency). Order matters: `["science", "no-latex"]`,
never the reverse.

`scienceplots` registers its bundled styles into `plt.style.library` as an IMPORT SIDE EFFECT
(see its own `__init__.py`) -- the `import scienceplots` below is required even though nothing
in this module calls it by name.

Every figure-producing function in `winder.plotting.latents` calls `apply_style()` itself,
rather than requiring the caller to remember to (module docstring there explains why); this
module's own contract is just "what the style is" and "how to check a figure honours the
title/caption convention".
"""

import matplotlib.pyplot as plt
import scienceplots  # type: ignore[import-untyped]  # noqa: F401 -- registers styles on import

__all__ = [
    "SCIENCEPLOTS_STYLE",
    "EXTRA_RCPARAMS",
    "apply_style",
    "assert_no_baked_in_title_or_caption",
]

#: Applied in this order: "science" first (the base look), "no-latex" second so its
#: `text.usetex: False` / `font.serif: STIXGeneral` override "science"'s own LaTeX default.
SCIENCEPLOTS_STYLE = ["science", "no-latex"]

#: `pdf.fonttype: 42` embeds real, editable glyph outlines in PDF output (Type 42 / TrueType),
#: instead of matplotlib's default Type 3 bitmap fonts -- the latter renders illegibly small text
#: in some PDF viewers and cannot be edited/re-flowed by a publisher's typesetting tools.
EXTRA_RCPARAMS = {"pdf.fonttype": 42}


def apply_style() -> None:
    """Apply the `scienceplots` `science`+`no-latex` style plus `EXTRA_RCPARAMS`, globally.

    Mutates `matplotlib.pyplot`'s global rcParams (matplotlib has no other notion of "current
    style"). Idempotent: calling it twice in a row leaves rcParams exactly as calling it once
    would -- `plt.style.use` fully re-applies each named style's file rather than accumulating.
    """
    plt.style.use(SCIENCEPLOTS_STYLE)
    # matplotlib's own stub types RcParams.update's argument against a huge Literal union of
    # every known rc key; a plain `dict[str, int]` is correctly rejected as "could be any key
    # string", not just this module's known-valid one -- a real stub-strictness mismatch, not a
    # bug in this call.
    plt.rcParams.update(EXTRA_RCPARAMS)  # type: ignore[arg-type]


def assert_no_baked_in_title_or_caption(fig: plt.Figure) -> None:
    """Raise `AssertionError` if `fig` or any of its Axes carries a title or figure-level caption.

    The convention this enforces (CTO review, P5.5 design brief): axes are labelled
    (`set_xlabel`/`set_ylabel`), but any caption or per-panel/per-arm identity belongs in the
    manuscript's own typesetting or a legend entry -- never `ax.set_title`/`fig.suptitle`, which
    would bake text into the image file itself. Checks all three title locations matplotlib
    supports per Axes (`center`, `left`, `right`) since `set_title(loc=...)` can set any of them
    independently of the other two.
    """
    for ax in fig.axes:
        for loc in ("center", "left", "right"):
            title = ax.get_title(loc=loc)
            if title != "":
                raise AssertionError(f"Axes {ax!r} has a non-empty {loc!r} title: {title!r}")
    # `fig.get_suptitle()` is the public accessor (matplotlib >= 3.4); the design brief's own
    # pseudocode said `fig._suptitle is None`, but that private attribute is None before any
    # suptitle is ever set and a Text object after -- `get_suptitle()` gives the same answer
    # ("" when unset) through documented API, verified this session on matplotlib 3.11.1.
    suptitle = fig.get_suptitle()
    if suptitle != "":
        raise AssertionError(f"Figure has a non-empty suptitle: {suptitle!r}")
