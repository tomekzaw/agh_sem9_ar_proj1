import numpy as np
import pytest

from utils import G, calculate_interaction, calculate_interactions

ci = calculate_interaction


def isclose(a: np.array, b: np.array) -> bool:
    return np.isclose(a, b, rtol=1e-14, atol=0).all()


def test_calculate_interaction() -> None:
    star_i = _, xi, yi, zi = np.random.random(4)
    star_j = _, xj, yj, zj = np.random.random(4)

    denominator = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2) ** 3

    expected = np.array([
        G * (xj - xi) / denominator,
        G * (yj - yi) / denominator,
        G * (zj - zi) / denominator,
    ])

    actual = calculate_interaction(star_i, star_j)

    assert isclose(expected, actual)


@pytest.mark.parametrize('use_symmetry', [True, False])
def test_calculate_interactions(use_symmetry: bool) -> None:
    stars_own = star0, star1, star2, star3, star4 = np.random.rand(5, 4)
    stars_buffer = starA, starB, starC = np.random.rand(3, 4)

    expected = np.array([
        [ci(star0, starA), ci(star0, starB), ci(star0, starC)],
        [ci(star1, starA), ci(star1, starB), ci(star1, starC)],
        [ci(star2, starA), ci(star2, starB), ci(star2, starC)],
        [ci(star3, starA), ci(star3, starB), ci(star3, starC)],
        [ci(star4, starA), ci(star4, starB), ci(star4, starC)],
    ])

    actual = calculate_interactions(stars_own, stars_buffer, same=False, use_symmetry=use_symmetry)

    assert isclose(expected, actual)


@pytest.mark.parametrize('same, use_symmetry', [(True, True), (True, False), (False, False)])
def test_calculate_interactions_own(same: bool, use_symmetry: bool) -> None:
    stars = star0, star1, star2, star3, star4 = np.random.rand(5, 4)

    zero3d = np.array([0, 0, 0])

    expected = np.array([
        [zero3d, ci(star0, star1), ci(star0, star2), ci(star0, star3), ci(star0, star4)],
        [ci(star1, star0), zero3d, ci(star1, star2), ci(star1, star3), ci(star1, star4)],
        [ci(star2, star0), ci(star2, star1), zero3d, ci(star2, star3), ci(star2, star4)],
        [ci(star3, star0), ci(star3, star1), ci(star3, star2), zero3d, ci(star3, star4)],
        [ci(star4, star0), ci(star4, star1), ci(star4, star2), ci(star4, star3), zero3d],
    ])

    actual = calculate_interactions(stars, stars, same=same, use_symmetry=use_symmetry)

    assert isclose(expected, actual)
