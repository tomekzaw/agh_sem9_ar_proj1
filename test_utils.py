import numpy as np
from utils import G, calculate_interaction, calculate_interactions_own, calculate_interactions_own_with_buffer


def isclose(a: np.array, b: np.array) -> bool:
    return np.isclose(a, b, rtol=1e-14, atol=0).all()


def test_calculate_interaction() -> None:
    star_i = Mi, xi, yi, zi = np.random.random(4)
    star_j = Mj, xj, yj, zj = np.random.random(4)

    denominator = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2) ** 3 + 1e-10

    expected = np.array([
        G * Mj * (xi - xj) / denominator,
        G * Mj * (yi - yj) / denominator,
        G * Mj * (zi - zj) / denominator,
    ])

    actual = calculate_interaction(star_i, star_j)

    assert isclose(expected, actual)


def test_calculate_interactions_own() -> None:
    stars_own = star0, star1, star2, star3, star4 = np.random.rand(5, 4)

    ci = calculate_interaction

    expected = np.array([
        ci(star0, star1) + ci(star0, star2) + ci(star0, star3) + ci(star0, star4),
        ci(star1, star0) + ci(star1, star2) + ci(star1, star3) + ci(star1, star4),
        ci(star2, star0) + ci(star2, star1) + ci(star2, star3) + ci(star2, star4),
        ci(star3, star0) + ci(star3, star1) + ci(star3, star2) + ci(star3, star4),
        ci(star4, star0) + ci(star4, star1) + ci(star4, star2) + ci(star4, star3),
    ])

    actual = calculate_interactions_own(stars_own)

    assert isclose(expected, actual)


def test_calculate_interactions_own_with_buffer() -> None:
    stars_own = star0, star1, star2, star3, star4 = np.random.rand(5, 4)
    stars_buffer = starA, starB, starC = np.random.rand(3, 4)

    ci = calculate_interaction

    expected = np.array([
        ci(star0, starA) + ci(star0, starB) + ci(star0, starC),
        ci(star1, starA) + ci(star1, starB) + ci(star1, starC),
        ci(star2, starA) + ci(star2, starB) + ci(star2, starC),
        ci(star3, starA) + ci(star3, starB) + ci(star3, starC),
        ci(star4, starA) + ci(star4, starB) + ci(star4, starC),
    ])

    actual = calculate_interactions_own_with_buffer(stars_own, stars_buffer)

    assert isclose(expected, actual)
