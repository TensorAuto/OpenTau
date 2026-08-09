#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from unittest.mock import patch

import numpy as np
import pytest

from opentau.datasets.compute_stats import (
    QUANTILE_STAT_NAMES,
    _assert_type_and_shape,
    aggregate_feature_stats,
    aggregate_stats,
    compute_episode_stats,
    estimate_num_samples,
    get_feature_stats,
    sample_images,
    sample_indices,
)


def mock_load_image_as_numpy(path, dtype, channel_first):
    return np.ones((3, 32, 32), dtype=dtype) if channel_first else np.ones((32, 32, 3), dtype=dtype)


@pytest.fixture
def sample_array():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_estimate_num_samples():
    assert estimate_num_samples(1) == 1
    assert estimate_num_samples(10) == 10
    assert estimate_num_samples(100) == 100
    assert estimate_num_samples(200) == 100
    assert estimate_num_samples(1000) == 177
    assert estimate_num_samples(2000) == 299
    assert estimate_num_samples(5000) == 594
    assert estimate_num_samples(10_000) == 1000
    assert estimate_num_samples(20_000) == 1681
    assert estimate_num_samples(50_000) == 3343
    assert estimate_num_samples(500_000) == 10_000


def test_sample_indices():
    indices = sample_indices(10)
    assert len(indices) > 0
    assert indices[0] == 0
    assert indices[-1] == 9
    assert len(indices) == estimate_num_samples(10)


@patch("opentau.datasets.compute_stats.load_image_as_numpy", side_effect=mock_load_image_as_numpy)
def test_sample_images(mock_load):
    image_paths = [f"image_{i}.jpg" for i in range(100)]
    images = sample_images(image_paths)
    assert isinstance(images, np.ndarray)
    assert images.shape[1:] == (3, 32, 32)
    assert images.dtype == np.uint8
    assert len(images) == estimate_num_samples(100)


def test_get_feature_stats_images():
    data = np.random.rand(100, 3, 32, 32)
    stats = get_feature_stats(data, axis=(0, 2, 3), keepdims=True)
    assert "min" in stats and "max" in stats and "mean" in stats and "std" in stats and "count" in stats
    np.testing.assert_equal(stats["count"], np.array([100]))
    assert stats["min"].shape == stats["max"].shape == stats["mean"].shape == stats["std"].shape


def test_get_feature_stats_axis_0_keepdims(sample_array):
    expected = {
        "min": np.array([[1, 2, 3]]),
        "max": np.array([[7, 8, 9]]),
        "mean": np.array([[4.0, 5.0, 6.0]]),
        "std": np.array([[2.44948974, 2.44948974, 2.44948974]]),
        "count": np.array([3]),
    }
    result = get_feature_stats(sample_array, axis=(0,), keepdims=True)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key])


def test_get_feature_stats_axis_1(sample_array):
    expected = {
        "min": np.array([1, 4, 7]),
        "max": np.array([3, 6, 9]),
        "mean": np.array([2.0, 5.0, 8.0]),
        "std": np.array([0.81649658, 0.81649658, 0.81649658]),
        "count": np.array([3]),
    }
    result = get_feature_stats(sample_array, axis=(1,), keepdims=False)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key])


def test_get_feature_stats_no_axis(sample_array):
    expected = {
        "min": np.array(1),
        "max": np.array(9),
        "mean": np.array(5.0),
        "std": np.array(2.5819889),
        "count": np.array([3]),
    }
    result = get_feature_stats(sample_array, axis=None, keepdims=False)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key])


def test_get_feature_stats_empty_array():
    array = np.array([])
    with pytest.raises(ValueError):
        get_feature_stats(array, axis=(0,), keepdims=True)


def test_get_feature_stats_single_value():
    array = np.array([[1337]])
    result = get_feature_stats(array, axis=None, keepdims=True)
    np.testing.assert_equal(result["min"], np.array(1337))
    np.testing.assert_equal(result["max"], np.array(1337))
    np.testing.assert_equal(result["mean"], np.array(1337.0))
    np.testing.assert_equal(result["std"], np.array(0.0))
    np.testing.assert_equal(result["count"], np.array([1]))


def test_compute_episode_stats():
    episode_data = {
        "observation.image": [f"image_{i}.jpg" for i in range(100)],
        "observation.state": np.random.rand(100, 10),
    }
    features = {
        "observation.image": {"dtype": "image"},
        "observation.state": {"dtype": "numeric"},
    }

    with patch("opentau.datasets.compute_stats.load_image_as_numpy", side_effect=mock_load_image_as_numpy):
        stats = compute_episode_stats(episode_data, features)

    assert "observation.image" in stats and "observation.state" in stats
    assert stats["observation.image"]["count"].item() == 100
    assert stats["observation.state"]["count"].item() == 100
    assert stats["observation.image"]["mean"].shape == (3, 1, 1)


def test_assert_type_and_shape_valid():
    valid_stats = [
        {
            "feature1": {
                "min": np.array([1.0]),
                "max": np.array([10.0]),
                "mean": np.array([5.0]),
                "std": np.array([2.0]),
                "count": np.array([1]),
            }
        }
    ]
    _assert_type_and_shape(valid_stats)


def test_assert_type_and_shape_invalid_type():
    invalid_stats = [
        {
            "feature1": {
                "min": [1.0],  # Not a numpy array
                "max": np.array([10.0]),
                "mean": np.array([5.0]),
                "std": np.array([2.0]),
                "count": np.array([1]),
            }
        }
    ]
    with pytest.raises(ValueError, match="Stats must be composed of numpy array"):
        _assert_type_and_shape(invalid_stats)


def test_assert_type_and_shape_invalid_shape():
    invalid_stats = [
        {
            "feature1": {
                "count": np.array([1, 2]),  # Wrong shape
            }
        }
    ]
    with pytest.raises(ValueError, match=r"Shape of 'count' must be \(1\)"):
        _assert_type_and_shape(invalid_stats)


def test_aggregate_feature_stats():
    stats_ft_list = [
        {
            "min": np.array([1.0]),
            "max": np.array([10.0]),
            "mean": np.array([5.0]),
            "std": np.array([2.0]),
            "count": np.array([1]),
        },
        {
            "min": np.array([2.0]),
            "max": np.array([12.0]),
            "mean": np.array([6.0]),
            "std": np.array([2.5]),
            "count": np.array([1]),
        },
    ]
    result = aggregate_feature_stats(stats_ft_list)
    np.testing.assert_allclose(result["min"], np.array([1.0]))
    np.testing.assert_allclose(result["max"], np.array([12.0]))
    np.testing.assert_allclose(result["mean"], np.array([5.5]))
    np.testing.assert_allclose(result["std"], np.array([2.318405]), atol=1e-6)
    np.testing.assert_allclose(result["count"], np.array([2]))


def test_aggregate_feature_stats_nan_tolerant_per_dim():
    """A NaN at one contributor's dim must not poison clean dims for other contributors.

    Regression: a single dataset's NaN min/max would poison every sample's
    normalize buffer because np.min/np.max propagate NaN. The fix uses
    np.nanmin/np.nanmax + per-dim NaN-aware weighted mean/variance.
    """
    nan = float("nan")
    stats_ft_list = [
        {  # contributor A: dim 0 is NaN, dim 1 is clean
            "min": np.array([nan, 2.0]),
            "max": np.array([nan, 12.0]),
            "mean": np.array([nan, 6.0]),
            "std": np.array([nan, 2.5]),
            "count": np.array([10]),
        },
        {  # contributor B: both dims clean
            "min": np.array([1.0, 1.5]),
            "max": np.array([10.0, 11.0]),
            "mean": np.array([5.0, 5.5]),
            "std": np.array([2.0, 2.0]),
            "count": np.array([10]),
        },
    ]
    result = aggregate_feature_stats(stats_ft_list)
    # dim 0: only B contributes (A is NaN at this dim)
    np.testing.assert_allclose(result["min"][0], 1.0)
    np.testing.assert_allclose(result["max"][0], 10.0)
    np.testing.assert_allclose(result["mean"][0], 5.0)
    np.testing.assert_allclose(result["std"][0], 2.0)
    # dim 1: weighted average of A and B (equal counts)
    np.testing.assert_allclose(result["min"][1], 1.5)
    np.testing.assert_allclose(result["max"][1], 12.0)
    np.testing.assert_allclose(result["mean"][1], 5.75)
    # count is unaffected by per-dim NaN masking
    np.testing.assert_allclose(result["count"], np.array([20]))


def test_aggregate_feature_stats_inf_masked_per_dim():
    """+/-Inf in a contributor's stats must be masked per-dim the same way NaN is.

    Naive ``np.nanmin`` / ``np.nanmax`` skip NaN but *not* Inf -- +Inf would still
    poison the aggregated max, -Inf the aggregated min, and ``np.where(np.isnan(...))``
    would leave them in the weighted-mean numerator. This regression locks in the
    ``~np.isfinite`` predicate so a contributor with +/-Inf is excluded per dim
    just like a NaN contributor.
    """
    inf = float("inf")
    stats_ft_list = [
        {  # contributor A: dim 0 fully Inf-poisoned, dim 1 clean
            "min": np.array([-inf, 2.0]),
            "max": np.array([inf, 12.0]),
            "mean": np.array([inf, 6.0]),
            "std": np.array([inf, 2.5]),
            "count": np.array([10]),
        },
        {  # contributor B: clean everywhere
            "min": np.array([1.0, 1.5]),
            "max": np.array([10.0, 11.0]),
            "mean": np.array([5.0, 5.5]),
            "std": np.array([2.0, 2.0]),
            "count": np.array([10]),
        },
    ]
    result = aggregate_feature_stats(stats_ft_list)
    # dim 0: only B contributes (A is Inf at every stat); naive nanmin/nanmax
    # would return -inf / +inf here without the ~np.isfinite mask.
    np.testing.assert_allclose(result["min"][0], 1.0)
    np.testing.assert_allclose(result["max"][0], 10.0)
    np.testing.assert_allclose(result["mean"][0], 5.0)
    np.testing.assert_allclose(result["std"][0], 2.0)
    # dim 1: both contribute (A's stats are finite at this dim).
    np.testing.assert_allclose(result["min"][1], 1.5)
    np.testing.assert_allclose(result["max"][1], 12.0)
    np.testing.assert_allclose(result["mean"][1], 5.75)
    assert np.isfinite(result["min"]).all()
    assert np.isfinite(result["max"]).all()
    assert np.isfinite(result["mean"]).all()
    assert np.isfinite(result["std"]).all()


def test_aggregate_feature_stats_all_nan_dim_stays_nan():
    """If every contributor is NaN at a dim, the result is NaN there (not silently clean).

    Lets downstream loaders surface the case rather than masking it.
    """
    nan = float("nan")
    stats_ft_list = [
        {
            "min": np.array([nan, 2.0]),
            "max": np.array([nan, 12.0]),
            "mean": np.array([nan, 6.0]),
            "std": np.array([nan, 2.5]),
            "count": np.array([10]),
        },
        {
            "min": np.array([nan, 1.5]),
            "max": np.array([nan, 11.0]),
            "mean": np.array([nan, 5.5]),
            "std": np.array([nan, 2.0]),
            "count": np.array([10]),
        },
    ]
    result = aggregate_feature_stats(stats_ft_list)
    assert np.isnan(result["min"][0])
    assert np.isnan(result["max"][0])
    assert np.isnan(result["mean"][0])
    assert np.isnan(result["std"][0])
    np.testing.assert_allclose(result["min"][1], 1.5)
    np.testing.assert_allclose(result["max"][1], 12.0)


def test_aggregate_stats():
    all_stats = [
        {
            "observation.image": {
                "min": [1, 2, 3],
                "max": [10, 20, 30],
                "mean": [5.5, 10.5, 15.5],
                "std": [2.87, 5.87, 8.87],
                "count": 10,
            },
            "observation.state": {"min": 1, "max": 10, "mean": 5.5, "std": 2.87, "count": 10},
            "extra_key_0": {"min": 5, "max": 25, "mean": 15, "std": 6, "count": 6},
        },
        {
            "observation.image": {
                "min": [2, 1, 0],
                "max": [15, 10, 5],
                "mean": [8.5, 5.5, 2.5],
                "std": [3.42, 2.42, 1.42],
                "count": 15,
            },
            "observation.state": {"min": 2, "max": 15, "mean": 8.5, "std": 3.42, "count": 15},
            "extra_key_1": {"min": 0, "max": 20, "mean": 10, "std": 5, "count": 5},
        },
    ]

    expected_agg_stats = {
        "observation.image": {
            "min": [1, 1, 0],
            "max": [15, 20, 30],
            "mean": [7.3, 7.5, 7.7],
            "std": [3.5317, 4.8267, 8.5581],
            "count": 25,
        },
        "observation.state": {
            "min": 1,
            "max": 15,
            "mean": 7.3,
            "std": 3.5317,
            "count": 25,
        },
        "extra_key_0": {
            "min": 5,
            "max": 25,
            "mean": 15.0,
            "std": 6.0,
            "count": 6,
        },
        "extra_key_1": {
            "min": 0,
            "max": 20,
            "mean": 10.0,
            "std": 5.0,
            "count": 5,
        },
    }

    # cast to numpy
    for ep_stats in all_stats:
        for fkey, stats in ep_stats.items():
            for k in stats:
                stats[k] = np.array(stats[k], dtype=np.int64 if k == "count" else np.float32)
                if fkey == "observation.image" and k != "count":
                    stats[k] = stats[k].reshape(3, 1, 1)  # for normalization on image channels
                else:
                    stats[k] = stats[k].reshape(1)

    # cast to numpy
    for fkey, stats in expected_agg_stats.items():
        for k in stats:
            stats[k] = np.array(stats[k], dtype=np.int64 if k == "count" else np.float32)
            if fkey == "observation.image" and k != "count":
                stats[k] = stats[k].reshape(3, 1, 1)  # for normalization on image channels
            else:
                stats[k] = stats[k].reshape(1)

    results = aggregate_stats(all_stats)

    for fkey in expected_agg_stats:
        np.testing.assert_allclose(results[fkey]["min"], expected_agg_stats[fkey]["min"])
        np.testing.assert_allclose(results[fkey]["max"], expected_agg_stats[fkey]["max"])
        np.testing.assert_allclose(results[fkey]["mean"], expected_agg_stats[fkey]["mean"])
        np.testing.assert_allclose(
            results[fkey]["std"], expected_agg_stats[fkey]["std"], atol=1e-04, rtol=1e-04
        )
        np.testing.assert_allclose(results[fkey]["count"], expected_agg_stats[fkey]["count"])


def _base_stats(count: int = 10) -> dict[str, np.ndarray]:
    """Non-quantile stats every contributor in the quantile tests shares."""
    return {
        "min": np.array([-8.0]),
        "max": np.array([8.0]),
        "mean": np.array([0.0]),
        "std": np.array([1.0]),
        "count": np.array([count]),
    }


def _with_quantiles(count: int, scale: float) -> dict[str, np.ndarray]:
    """`_base_stats` plus the full quantile set, spread symmetrically around 0."""
    stats = _base_stats(count)
    for q_name, offset in zip(QUANTILE_STAT_NAMES, (-2.0, -1.0, 0.0, 1.0, 2.0), strict=True):
        stats[q_name] = np.array([offset * scale])
    return stats


def test_aggregate_feature_stats_quantiles_weighted_mean():
    """q01/q99 aggregate as the count-weighted mean of contributor quantiles."""
    stats_ft = [
        {
            "min": np.array([0.0]),
            "max": np.array([10.0]),
            "mean": np.array([5.0]),
            "std": np.array([1.0]),
            "count": np.array([10]),
            "q01": np.array([1.0]),
            "q99": np.array([9.0]),
        },
        {
            "min": np.array([0.0]),
            "max": np.array([10.0]),
            "mean": np.array([5.0]),
            "std": np.array([1.0]),
            "count": np.array([30]),
            "q01": np.array([3.0]),
            "q99": np.array([5.0]),
        },
    ]
    agg = aggregate_feature_stats(stats_ft)
    # weighted by counts 10:30 -> q01 = (1*10 + 3*30)/40 = 2.5 ; q99 = (9*10 + 5*30)/40 = 6.0
    np.testing.assert_allclose(agg["q01"], [2.5])
    np.testing.assert_allclose(agg["q99"], [6.0])


def test_aggregate_feature_stats_full_quantile_set_aggregates(caplog):
    """(a) Every contributor has every quantile -> all five land in the aggregate, no warning.

    Pins the inner quantiles too: the loop used to cover only q01/q99, so q10/q50/q90 — which
    fleet stats now carry and `_to_standard_data_format` already pads — were silently dropped
    from every aggregated norm head.
    """
    stats_ft = [_with_quantiles(count=10, scale=1.0), _with_quantiles(count=30, scale=2.0)]
    with caplog.at_level(logging.WARNING):
        agg = aggregate_feature_stats(stats_ft)
    assert set(QUANTILE_STAT_NAMES) <= set(agg)
    # count-weighted 10:30 of scale 1 and 2 -> effective scale 1.75
    np.testing.assert_allclose(agg["q01"], [-3.5])
    np.testing.assert_allclose(agg["q10"], [-1.75])
    np.testing.assert_allclose(agg["q50"], [0.0])
    np.testing.assert_allclose(agg["q90"], [1.75])
    np.testing.assert_allclose(agg["q99"], [3.5])
    assert not [r for r in caplog.records if "dropping" in r.getMessage()]


def test_aggregate_feature_stats_no_quantiles_no_keys(caplog):
    """(b) No contributor has quantiles -> keys absent, and no warning noise.

    The overwhelmingly common case today (nothing in the mixture is migrated yet); warning on it
    would fire on nearly every job and train operators to ignore the message that matters.
    """
    stats_ft = [_base_stats(count=5), _base_stats(count=7)]
    with caplog.at_level(logging.WARNING):
        agg = aggregate_feature_stats(stats_ft)
    assert not set(QUANTILE_STAT_NAMES) & set(agg)
    assert caplog.records == []


@pytest.mark.parametrize("migrated_index", [0, 1])
def test_aggregate_feature_stats_partial_quantiles_skips_and_warns(migrated_index, caplog):
    """(c) Partial coverage in BOTH directions -> quantiles dropped, warned, never raised.

    The failure this replaces was symmetric: the old KeyError fired whether the migrated dataset
    came first or second, killing any job whose mixture straddled the incremental fleet migration
    before it even loaded a checkpoint. Parametrizing the position is what makes this a pin —
    a fix that only handled "new dataset last" would pass a single-ordering test.

    Backfilling is still refused: `min` sits however far outside the 1st percentile the tail
    reaches, so pooling it with a true q01 would widen the band by an outlier-driven amount —
    exactly the sensitivity QUANTILE exists to remove. Declining to publish a quantile at all is
    the only safe option, and non-QUANTILE jobs over the same mixture are unaffected.
    """
    stats_ft = [_base_stats(count=10), _base_stats(count=10)]
    stats_ft[migrated_index] = _with_quantiles(count=10, scale=1.0)
    unmigrated_index = 1 - migrated_index

    with caplog.at_level(logging.WARNING):
        agg = aggregate_feature_stats(
            stats_ft, contributor_names=["TensorAuto/migrated", "TensorAuto/legacy"]
        )

    # No exception, and the non-quantile stats still aggregate normally.
    assert not set(QUANTILE_STAT_NAMES) & set(agg)
    np.testing.assert_allclose(agg["mean"], [0.0])
    np.testing.assert_allclose(agg["count"], [20])

    # Exactly one warning, naming the unmigrated contributor and every dropped quantile.
    warnings_logged = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_logged) == 1
    message = warnings_logged[0]
    assert ["TensorAuto/migrated", "TensorAuto/legacy"][unmigrated_index] in message
    assert ["TensorAuto/migrated", "TensorAuto/legacy"][migrated_index] not in message
    for q_name in QUANTILE_STAT_NAMES:
        assert q_name in message
    assert "1/2 contributors" in message


def test_aggregate_feature_stats_partial_quantiles_groups_by_missing_set(caplog):
    """Quantiles missing from *different* contributors warn separately, not as one blurred line.

    A dataset migrated to q01/q99 only, alongside one with the full set, drops q10/q50/q90 for a
    different reason than a dataset with no quantiles at all drops q01/q99 — telling the operator
    which recompute fixes which gap requires keeping the two groups apart.
    """
    full = _with_quantiles(count=10, scale=1.0)
    outer_only = _base_stats(count=10) | {"q01": np.array([-2.0]), "q99": np.array([2.0])}
    none_at_all = _base_stats(count=10)

    with caplog.at_level(logging.WARNING):
        agg = aggregate_feature_stats(
            [full, outer_only, none_at_all], contributor_names=["full", "outer_only", "none_at_all"]
        )

    assert not set(QUANTILE_STAT_NAMES) & set(agg)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 2
    # q01/q99 are missing from one contributor; q10/q50/q90 from two.
    outer = next(m for m in messages if "'q01'" in m)
    inner = next(m for m in messages if "'q10'" in m)
    assert "1/3 contributors" in outer and "none_at_all" in outer and "outer_only" not in outer
    assert "2/3 contributors" in inner and "none_at_all" in inner and "outer_only" in inner


def test_aggregate_feature_stats_partial_quantiles_falls_back_to_indices(caplog):
    """Without contributor names the warning still points somewhere: the list index."""
    with caplog.at_level(logging.WARNING):
        aggregate_feature_stats([_with_quantiles(count=10, scale=1.0), _base_stats(count=10)])
    message = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "indices [1]" in message


def test_aggregate_feature_stats_rejects_misaligned_contributor_names():
    """A names list that isn't parallel would name the wrong dataset — reject it up front."""
    with pytest.raises(ValueError, match="parallel"):
        aggregate_feature_stats([_base_stats(), _base_stats()], contributor_names=["only-one"])


def test_aggregate_stats_names_the_offending_dataset_per_feature(caplog):
    """`aggregate_stats` forwards names/context per feature, keeping both aligned to the filter.

    `state` is present on both contributors while `actions` is present on only one, so the two
    features aggregate over different contributor subsets. The names (and weights) must be
    filtered by the same positions, or the warning would accuse whichever dataset happens to sit
    at that index in the unfiltered list.
    """
    migrated = {"state": _with_quantiles(count=10, scale=1.0), "actions": _with_quantiles(10, 1.0)}
    legacy = {"state": _base_stats(count=10)}

    with caplog.at_level(logging.WARNING):
        agg = aggregate_stats(
            [migrated, legacy],
            weights=[1.0, 3.0],
            contributor_names=["TensorAuto/migrated", "TensorAuto/legacy"],
            context="norm head 'so101|joint_position'",
        )

    # `actions` has a single contributor, so its quantiles survive; `state` loses them.
    assert set(QUANTILE_STAT_NAMES) <= set(agg["actions"])
    assert not set(QUANTILE_STAT_NAMES) & set(agg["state"])
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 1
    assert "TensorAuto/legacy" in messages[0]
    assert "norm head 'so101|joint_position'" in messages[0] and "feature 'state'" in messages[0]
