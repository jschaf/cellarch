"""
TODO: model data placement as a distribution.  Currently uniform.
TODO: model mean time to repair as a distribution
TODO: model mean time to failure as a distribution

𝑛 - number of machines. A machine exists in exactly 1 cell.
𝑐 - the number of cells. A cell is a group of machines. Assume each cell contains the same number of machines.
𝑓 - mean time to failure of an individual machine.
𝑟 - the number of replicas for each piece of data. If data is replicated twice, it exists on 2 separate machines in the same cell.
𝑑 - the number of pieces of data in the system. Data is uniformly distributed across cells and each cell uniformly distributes data across machines.
𝑡 - the recovery time for a machine after it fails.
"""
# %% Imports
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, FrozenSet, Tuple, NewType, Callable

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import datetime
import random

# %%

style = sns.set_style('whitegrid')
# Make plots work in IntelliJ
plt.interactive(False)

# %%
NUM_SIMULATIONS = 10
SIMULATION_DURATION = datetime.timedelta(days=365 * 4)

NUM_MACHINES = 100
NUM_PARTITIONS = 20
MEAN_TIME_TO_FAILURE = datetime.timedelta(days=30 * 6)
TIME_TO_REPAIR = datetime.timedelta(days=3)
NUM_REPLICAS = 2
NUM_DATA: int = int(1e5)
RECOVERY_TIME = datetime.timedelta(days=7)

# A one-dimensional, random distribution that takes a int size
# and returns samples from the distribution.  For example, here's
# a RandomDist from a uniform distribution with a mean of 5 and
# stddev of 2:
#
#     lambda size: np.random.normal(5, 2, size)
RandomDist1d = NewType('RandomDist', Callable[[int], np.ndarray])


def to_hours(delta: datetime.timedelta) -> int:
    """Converts a timedelta into the number of hours.

    Timedelta normalizes the delta into days, seconds and microseconds so
    'delta.hours' only returns the hours passed into the constructor."""
    return int(delta.total_seconds()) // 3600


def time_to_failure_exp_dist(
        num_machines: int,
        mttf: datetime.timedelta,
        sim_duration: datetime.timedelta) -> np.ndarray:
    """Generates a two-dimensional array of increasing failure times of
    integer hours from an exponential distribution.

    The returned ndarray has `num_machines` rows.  Each row represents the
    start of failures for a single machine.  Each row has the same number
    of columns.  The last value of each column is guaranteed to exceed the
    simulation duration.
    """
    mttf_hours = to_hours(mttf)
    sim_hours = to_hours(sim_duration)
    # Use a generous fudge factor so we only have to drop columns, not add
    # new columns to ensure the last column > min_duration.
    fudge_factor = 3
    num_cols = (sim_hours // mttf_hours) * fudge_factor

    # Add 1 to avoid having duplicate failure times if we get 0 from the
    # exponential distribution.
    all_failures = np.rint(np.random.exponential(mttf_hours, num_machines * num_cols)) + 1
    reshaped = np.reshape(all_failures, (num_machines, num_cols))
    summed = np.cumsum(reshaped, 1)

    # Drop excess columns
    col_mins = np.min(summed, axis=0)
    cols_exceeding_sim_duration = np.nonzero(col_mins > sim_hours)[0]
    assert cols_exceeding_sim_duration.size > 0, (
        'Not enough samples to exceed sim duration for all columns')
    # Add 1 because we want to keep the last column.
    cols_to_keep = cols_exceeding_sim_duration[0] + 1
    return np.delete(summed, np.s_[cols_to_keep:], axis=1)


def time_to_repair_uniform_dist(
        size: int,
        time_to_repair: datetime.timedelta) -> np.ndarray:
    """Returns an ndarray of integer hours of the times to repair drawn from
    a normal distribution with a mean of time_to_repair.

    Draws size samples. The min value is 1.
    """
    mean = to_hours(time_to_repair)
    # Choose a reasonable stddev.
    stddev = max(to_hours(time_to_repair) // 3, 1)
    normal_dist = np.random.normal(mean, scale=stddev, size=size)
    return np.clip(np.rint(normal_dist), a_min=1, a_max=None)


time_to_failure_exp_dist(5, datetime.timedelta(hours=10), datetime.timedelta(hours=50))
# %%%


# time_to_repair_uniform_dist(5)
np.nonzero(np.arange(5) > 3)


# %%
@dataclass
class Partition:
    """A partition is a group of machines.
    Each partition contains the same number of machines.
    """
    machines: List[int]


def days(n: int):
    return datetime.timedelta(days=n)


assert num_machines % num_partitions == 0, "The number of machines must be divisible by the number of partitions."
machines_per_part = num_machines // num_partitions
assert machines_per_part >= num_replicas, "The number of machines per partition must exceed the replication factor."

machine_locs = [list(range(n * machines_per_part, n * machines_per_part + machines_per_part)) for n in
                range(num_partitions)]
PARTITIONS = [Partition(locs) for locs in machine_locs]


def distribute_data(n: int, replicas: int, parts: List[Partition]) -> Dict[FrozenSet[int], List[int]]:
    """Places n pieces of data with the following constraints:

    - Each piece of data is replicated `replicas` times.
    - Data is replicated on machines in the same partition.
    - Data is replicated on different machines within a partition.
    """
    data_index = defaultdict(list)
    for i in range(n):
        partition = random.randrange(0, num_partitions)
        population = parts[partition].machines
        machines = frozenset(sorted(random.sample(population, k=replicas)))
        data_index[machines].append(i)

    return data_index


DATA_DIST = distribute_data(num_data, replicas=num_replicas, parts=PARTITIONS)

# %%
# Check distribution of data across machines
counts_per_machine = pd.Series(len(v) for v in DATA_DIST.values())
assert counts_per_machine.count() == num_machines * num_replicas
counts_per_machine.describe()

# %%
# Check distribution of data across machine pairs
pd.Series(DATA_DIST.keys()).describe()


# %%

@dataclass
class Overlap:
    """An overlap is whenever two machines are down at the same time.

    The machines are not necessarily related.
    """
    machines: FrozenSet[int]
    start_hour: int
    end_hour: int


@dataclass
class Outage:
    """An outage is any period of time where two machines that share data are
    down at the same time.

    An outage is partial if the number of machines < NUM_REPLICAS.  An outage
    is full if machines == NUM_REPLICAS.
    """
    machines: FrozenSet[int]
    start_hour: int
    end_hour: int
    data: List[int]


# %%


def gen_machine_failure_starts(
        num_machines: int,
        time_to_failure_dist: RandomDist1d,
) -> np.ndarray:
    """Generates a two-dimensional array of increasing failure times of
    integer hours from an exponential distribution.

    The returned ndarray has `num_machines` rows.  Each row has the same
    number of columns.  Each row is the cumulative sum (cumsum) of samples
    from an exponential distribution using the mean time to failure, `mttf`.
     The last value of each column is guaranteed to exceed min_duration.
    """
    assert num_machines > 0
    all_failures = time_to_failure_dist(num_machines)
    num_cols = len(all_failures[0])
    reshaped = np.reshape(all_failures, (num_machines, num_cols))
    summed = np.cumsum(reshaped, 1)
    last_col = summed[:, -1]

    # Drop excess columns
    cols_to_keep = 0
    for i in range(num_cols):
        cols_to_keep += 1
        min_val = np.min(summed[:, i])
        if min_val >= duration_hours:
            break
    return np.delete(summed, np.s_[cols_to_keep:], axis=1)


num_machines = 2
ttf = datetime.timedelta(hours=4)
min_duration = datetime.timedelta(hours=30)
a = gen_machine_failure_starts(num_machines, ttf, min_duration)
print(a)


def gen_machine_down_times(
        failure_starts: np.ndarray,
        down_time_dist: RandomDist1d) -> np.ndarray:
    """Generates a ragged 2d array of 2-element [start_time, end_time] pairs
    from failure_starts indexed the machine number.

    Each row represents the failures for a single machine.  Each column within
    the row contains two-element array of [start_time, end_time].

    The returned ndarray is transformed by:

    1.  Replacing each cell in the two-dimensional failure_starts ndarray
        with two-element array of the failure start time and the failure
        end time given by (start_time + down_time).

    2.  Coalescing overlapping and immediately adjacent failures into a single
        two dimensional array.  For example, [[1, 5], [5, 10]] is coalesced
        into [[1, 10]].
    """

    repair_times = down_time_dist(failure_starts.size).reshape(failure_starts.shape)
    failure_ends = failure_starts + repair_times
    # TODO: coalesce adjacent and overlapping down-time ranges
    # It's unlikely enough for large mttf that probably okay to
    # ignore
    return np.dstack((failure_starts, failure_ends))


gen_machine_down_times(a, time_to_repair_dist)

# %%

num_machines = 2
ttf = datetime.timedelta(hours=4)
min_duration = datetime.timedelta(hours=30)
gen_machine_failure_starts(num_machines, ttf, min_duration)

fails = np.arange(20).reshape((2, 10))
print(fails)
repairs = time_to_repair_dist(20).reshape((2, 10))
print(repairs)
end_times = fails + repairs
np.dstack((fails, end_times))


# %%
# We want a series for each machine of when it fails.
# TODO: we want multiple failures for a machine in some period

# Here's the first failure for each machine.
def simulate() -> List[Outage]:
    first_failures = np.around(np.random.exponential(time_to_failure.days * 24, num_machines))
    first_failures.sort()
    # TODO: we can use a distribution for repair time here.
    repair_time = first_failures + time_to_repair.days * 24
    machines = np.arange(num_machines)
    np.random.shuffle(machines)
    machine_fail_repair = np.transpose(np.vstack((machines, first_failures, repair_time)))

    # Find all sets of overlaps.
    overlaps = []
    for i in range(1, num_machines):
        hi_machine, hi_start, hi_end = machine_fail_repair[i]
        for j in reversed(range(i - 1)):
            lo_machine, lo_start, lo_end = machine_fail_repair[j]
            if lo_end <= hi_start:
                break
            else:
                overlaps.append(
                    Overlap(
                        machines=frozenset([int(lo_machine), int(hi_machine)]),
                        start_hour=hi_start,
                        end_hour=lo_end))

    outages = []
    for overlap in overlaps:
        if overlap.machines in DATA_DIST:
            outages.append(Outage(
                machines=overlap.machines,
                start_hour=overlap.start_hour,
                end_hour=overlap.end_hour,
                data=DATA_DIST[overlap.machines]
            ))
    return outages


# %%
num_outages = []
for _ in range(100):
    outages = simulate()
    num_outages.append(len(outages))
num_outages


# %% Simulate failure
def fail_machines(duration: datetime.timedelta, mttf: datetime.timedelta):
    mttf_hours = mttf.total_seconds() / (60 * 60)
    fail_prob = 1 / mttf_hours
    return np.random.binomial(1, fail_prob, duration.days * 24)


def should_fail(time_to_fail: datetime.timedelta) -> bool:
    """Returns true if a failure should occur."""
    return (1.0 / time_to_fail.days) < random.random()


# %%
sum(fail_machines(duration=days(30 * 6), mttf=days(10)) == 1)


# %% Simulate day

def simulate_hour():
    pass


# %%

# The number of days before you get 1000 successes in a row with a 10% chance.
np.random.negative_binomial(1000, 0.1)
