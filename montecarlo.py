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
RandomDist1dFn = NewType('RandomDist', Callable[[int], np.ndarray])


def hours(n: int) -> datetime.timedelta:
    return datetime.timedelta(hours=n)


def total_hours(delta: datetime.timedelta) -> int:
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
    mttf_hours = total_hours(mttf)
    sim_hours = total_hours(sim_duration)
    # Use a generous fudge factor so we only have to drop columns, not add
    # new columns to ensure the last column > min_duration.
    fudge_factor = 5
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
    mean = total_hours(time_to_repair)
    # Choose a reasonable stddev.
    stddev = max(total_hours(time_to_repair) // 3, 1)
    normal_dist = np.random.normal(mean, scale=stddev, size=size)
    return np.clip(np.rint(normal_dist), a_min=1, a_max=None)


time_to_failure_exp_dist(5, mttf=hours(10), sim_duration=hours(50))


# %%%

@dataclass(frozen=True, order=True)
class Outage:
    """An outage is any period of time where a machine is down."""
    # Order is important, we want to sort by hours first.
    start_hour: int
    end_hour: int
    machine: int


def gen_machine_outages(
        failure_starts: np.ndarray,
        time_to_repair_dist: RandomDist1dFn) -> List[Outage]:
    """Generates a list of outages from failure starts using a distribution to
    model the time to repair..
    """
    num_rows = failure_starts.shape[0]
    num_cols = failure_starts.shape[1]
    machine_idxs = np.repeat(np.arange(num_rows), num_cols).reshape(failure_starts.shape)

    repair_times = time_to_repair_dist(failure_starts.size).reshape(failure_starts.shape)
    failure_ends = failure_starts + repair_times
    start_end_idxs = np.dstack((machine_idxs, failure_starts, failure_ends))
    # Flatten all but the last dimension to get [[start, end, machine], ...]
    # https://stackoverflow.com/a/26553855/30900
    reshaped = start_end_idxs.reshape(-1, start_end_idxs.shape[-1])

    # TODO: We should merge overlapping and adjacent outages for a machine.
    # Not a huge concern for a large mttf.
    return [
        Outage(machine=int(machine), start_hour=int(start), end_hour=int(end))
        for machine, start, end in reshaped
    ]


num_machines = 2
sim_duration = hours(15)
failure_starts = time_to_failure_exp_dist(
    num_machines,
    mttf=hours(10),
    sim_duration=sim_duration)

mean_time_to_repair = hours(2)
gen_machine_outages(
    failure_starts,
    lambda size: time_to_repair_uniform_dist(size, mean_time_to_repair),
)
# %%
np.repeat(np.arange(3), 11)


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


# %%


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
