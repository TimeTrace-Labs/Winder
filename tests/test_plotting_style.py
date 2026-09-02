import matplotlib.pyplot as plt
import pytest

from winder.plotting.style import apply_style, assert_no_baked_in_title_or_caption


def test_apply_style_sets_scienceplots_no_latex_rcparams() -> None:
    """`apply_style()` must actually land the `no-latex` override, not just `science`'s own
    (LaTeX-dependent) defaults -- a style-application order bug would leave `text.usetex` True."""
    apply_style()
    assert plt.rcParams["text.usetex"] is False
    # matplotlib returns font.family as a list even for a single value -- asserting the bare
    # string here would pass by accident on some matplotlib versions and fail on this one.
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["pdf.fonttype"] == 42


def test_assert_no_baked_in_title_or_caption_passes_on_a_clean_figure() -> None:
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    assert_no_baked_in_title_or_caption(fig)  # must not raise
    plt.close(fig)


def test_assert_no_baked_in_title_or_caption_catches_an_axes_title() -> None:
    fig, ax = plt.subplots()
    ax.set_title("should not be here")
    with pytest.raises(AssertionError, match="title"):
        assert_no_baked_in_title_or_caption(fig)
    plt.close(fig)


def test_assert_no_baked_in_title_or_caption_catches_a_left_or_right_axes_title() -> None:
    """`set_title(loc=...)` sets an independent Text object per location -- the checker must
    not only look at the default (center) title."""
    fig, ax = plt.subplots()
    ax.set_title("should not be here", loc="left")
    with pytest.raises(AssertionError, match="'left'"):
        assert_no_baked_in_title_or_caption(fig)
    plt.close(fig)


def test_assert_no_baked_in_title_or_caption_catches_a_figure_suptitle() -> None:
    fig, _ax = plt.subplots()
    fig.suptitle("should not be here either")
    with pytest.raises(AssertionError, match="suptitle"):
        assert_no_baked_in_title_or_caption(fig)
    plt.close(fig)
