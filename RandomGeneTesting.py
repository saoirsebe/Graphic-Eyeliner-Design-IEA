import pytest
import RandomGene


# Our fake random function for testing: returns the mean.
def fake_random_normal_within_range(mean, std_dev, value_range):
    return mean


@pytest.mark.parametrize(
    "segment_number, branch_length, max_segments, base_mean, std_dev, value_range, expected",
    [
        # Test 1: Early segment, shallow branch.
        # global_decay_factor = (1 - 1/100)**1.6 ≈ 0.99**1.6 ≈ 0.984,
        # branch_decay_factor = 1 - (1/5) = 0.8,
        # decay_factor ≈ 0.984 * 0.8 = 0.7872,
        # mean ≈ 4 * 0.7872 ≈ 3.1488, round(3.1488) = 3.
        (1, 1, 100, 4, 1, (0, 10), 3),

        # Test 2: Late segment: global_decay_factor becomes 0.
        (100, 1, 100, 4, 1, (0, 10), 0),

        # Test 3: Deep branch: branch_decay_factor becomes 0 (branch_length == average_branch_length, here 5).
        (1, 5, 100, 4, 1, (0, 10), 0),

        # Test 4: Mid-range values.
        # For segment_number=50, branch_length=2, max_segments=100, base_mean=6:
        # global_decay_factor = (1 - 50/100)**1.6 = (0.5)**1.6 ≈ 0.330,
        # branch_decay_factor = 1 - (2/5) = 0.6,
        # decay_factor ≈ 0.330 * 0.6 = 0.198,
        # mean ≈ 6 * 0.198 ≈ 1.188, round(1.188) = 1.
        (50, 2, 100, 6, 1, (0, 10), 1),

        # Test 5: Early segment but deeper branch.
        # For segment_number=1, branch_length=3, max_segments=100, base_mean=8:
        # global_decay_factor ≈ (1 - 1/100)**1.6 ≈ 0.984,
        # branch_decay_factor = 1 - (3/5) = 0.4,
        # decay_factor ≈ 0.984 * 0.4 ≈ 0.3936,
        # mean ≈ 8 * 0.3936 ≈ 3.1488, round(3.1488) = 3.
        (1, 3, 100, 8, 1, (0, 10), 3),
    ]
)
def test_n_of_children_decreasing_likelihood(segment_number, branch_length, max_segments, base_mean, std_dev,
                                             value_range, expected, monkeypatch):
    # Patch the random_normal_within_range in the RandomGeneTesting module.
    monkeypatch.setattr(RandomGene, "random_normal_within_range", fake_random_normal_within_range)

    result = RandomGene.n_of_children_decreasing_likelihood(segment_number, branch_length, max_segments,
                                                                   base_mean, std_dev, value_range)
    assert result == expected, (
        f"Expected {expected} children, got {result} for segment_number={segment_number}, "
        f"branch_length={branch_length}, base_mean={base_mean}"
    )