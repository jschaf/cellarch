import datetime

import hypothesis as hp
import pytest
from hypothesis import strategies as hps
import numpy as np

from montecarlo import gen_machine_failure_starts, normal_dist, hours


@hps.composite
def numpy_dists(draw):
    x = draw(hps.integers(min_value=1, max_value=100_000))
    return draw(hps.sampled_from([
        lambda size: np.random.normal(x, size=size),
        lambda size: np.random.exponential(x, size=size)]))


def machines_count():
    return hps.integers(min_value=1, max_value=100)


def sim_durations():
    return hps.timedeltas(
        min_value=datetime.timedelta(hours=1), max_value=datetime.timedelta(days=100_000))


@hp.given(machines_count(), hps.integers(min_value=1), sim_durations())
def test_gen_machine_failure_starts__normal_dist(num_machines, normal_mean, duration):
    gen_machine_failure_starts(
        num_machines,
        failure_dist=lambda size: normal_dist(hours(normal_mean), stddev=hours(1), size=size),
        sim_duration=duration)
    assert 1 == 1


@pytest.mark.parametrize(
    'path, query',
    [
        ('foo', 'f'),
        ('foo#id', 'f'),
        ('foo#id', 'fo'),
        ('foo#id', 'foo'),
        ('foo b', 'f b'),
    ]
)
def test_redis_client_search_works_for_tag_search(path, query):
    assert path == query
