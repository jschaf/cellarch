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
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, FrozenSet, NewType, Callable
import itertools

import numpy as np
import seaborn as sns
import matplotlib as plt
import datetime
import humanize

# %%

style = sns.set_style('whitegrid')
# Make plots work in IntelliJ
plt.interactive(False)

# %%

# A one-dimensional, random distribution that takes a int size
# and returns samples from the distribution.  For example, here's
# a RandomDist from a uniform distribution with a mean of 5 and
# stddev of 2:
#
#     lambda size: np.random.normal(5, 2, size)
RandomDist1dFn = NewType('RandomDist', Callable[[int], np.ndarray])


def hours(n: int) -> datetime.timedelta:
    return datetime.timedelta(hours=n)


def days(n: int):
    return datetime.timedelta(days=n)


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
    """An outage is the period of time where a machine is down."""
    # Order is important, we want to sort by hours first to ease finding
    # overlaps in outages.
    start_hour: int
    end_hour: int
    machine: int


def gen_machine_outages(
        num_machines: int,
        mttf: datetime.timedelta,
        time_to_repair_dist: RandomDist1dFn,
        sim_duration: datetime.timedelta) -> List[Outage]:
    """Generates a list of outages from failure starts using a distribution to
    model the time to repair.
    """
    failure_starts = time_to_failure_exp_dist(
        num_machines, mttf=mttf, sim_duration=sim_duration)
    repair_times = time_to_repair_dist(failure_starts.size).reshape(failure_starts.shape)
    failure_ends = failure_starts + repair_times
    start_ends = np.dstack((failure_starts, failure_ends))
    outages = []
    for row, failures in enumerate(start_ends):
        prev_end = float('-inf')
        for i, (start, end) in enumerate(failures):
            # If overlap, coalesce the outages into a single one.
            if start <= prev_end:
                prev_outage = outages[-1]
                outages[-1] = dataclasses.replace(prev_outage, end_hour=int(end))
            else:
                outages.append(
                    Outage(machine=row, start_hour=int(start), end_hour=int(end)))

            prev_end = end

    # Drop any outages outside the simulation duration.
    sim_hours = total_hours(sim_duration)
    return [o for o in outages if o.end_hour <= sim_hours]


# %%

@dataclass(frozen=True, order=True)
class OutageClique:
    """Outage cliques occur when multiple machines are down over the same time period."""
    # Order is important, we want to sort by hours first.
    start_hour: int
    end_hour: int
    machines: FrozenSet[int]

    @staticmethod
    def table_header():
        def under(word):
            return '=' * len(word)

        return (f'{"start":<25} {"duration":<20} {"machines"}\n'
                f'{under("start"):<25} {under("duration"):<20} '
                f'{under("machines")}')

    def format(self):
        duration_secs = (self.end_hour - self.start_hour) * 3600
        start = humanize.naturaldelta(self.start_hour * 3600)
        duration = humanize.naturaldelta(duration_secs)
        machines = ', '.join(str(m) for m in sorted(self.machines))
        return (f'{start:<25} '
                f'{duration:<20} '
                f'{machines}')

# %%

def find_outage_cliques(
        outages: List[Outage],
        clique_size: int) -> List[OutageClique]:
    """Finds all outage buddies where exactly cliqueSize machines were down
    in the same time period.

    The clique size will generally be the number of replicas.
    """
    if not outages:
        return []

    # Sort by start_hour, end_hour to enable a linear scan over outages.
    # We only need to look at the next clique_size outages for each outage.
    outages = sorted(outages)
    cliques = []

    for i in range(len(outages) - clique_size):
        outage = outages[i]
        clique_start = outage.start_hour
        clique_end = outage.end_hour

        for j in range(i + 1, i + clique_size):
            next_outage = outages[j]
            if next_outage.start_hour >= outage.end_hour:
                break
            # Found part of a clique.
            # Update the start because next_outage.start must be greater
            # than the prev outage.start because we sorted the list.
            clique_start = next_outage.start_hour
            # Use min because we're not sure which end is smaller.
            clique_end = min(clique_end, next_outage.end_hour)
        else:
            # The inner loop didn't break, so we have a clique.
            machines = set(x.machine for x in outages[i:i + clique_size])
            assert len(machines) == clique_size, 'Found same machine in clique'
            cliques.append(
                OutageClique(
                    start_hour=clique_start,
                    end_hour=clique_end,
                    machines=frozenset(sorted(machines))))
    return cliques


# %%

@dataclass(frozen=True)
class MachinePartition:
    """A partition is a group of machines.

    Each partition contains the same number of machines which should equal the
    number of replicas.
    """
    machines: FrozenSet[int]


def gen_partitions(num_machines: int, num_partitions: int) -> List[MachinePartition]:
    assert num_machines % num_partitions == 0, (
        'The number of machines must be divisible by the number of partitions.')
    m = num_machines // num_partitions
    return [MachinePartition(frozenset(range(n * m, n * m + m)))
            for n in range(num_partitions)]


def distribute_data(
        num_machines: int,
        num_partitions: int,
        num_replicas: int,
        num_data: int) -> Dict[FrozenSet[int], int]:
    """Uniformly distributes n pieces of data in partitions.

    - Each piece of data is replicated `replicas` times.
    - Data is replicated on machines in the same partition.
    - Data is replicated on different machines within a partition.
    """
    assert num_machines % num_partitions == 0, (
        'The number of machines must be divisible by the number of partitions.')
    machines_per_part = num_machines // num_partitions
    assert machines_per_part >= num_replicas, (
        'The number of replicas cannot exceed machines per partition')

    # Create partitions and cliques.
    partitions = gen_partitions(num_machines, num_partitions)
    cliques = []
    for partition in partitions:
        for combo in itertools.combinations(partition.machines, num_replicas):
            cliques.append(frozenset(combo))

    data_per_clique = num_data / len(cliques)
    assert data_per_clique > 1, 'must have at least 1 data per clique'

    # Probabilistically distribute data to avoid looping num_data times.
    stddev = 0.05
    data_dist = np.rint(np.random.normal(
        data_per_clique, data_per_clique * stddev, len(cliques)))
    counts = {}
    for clique, count in zip(cliques, data_dist):
        counts[clique] = count

    # Add or subtract data so it equals num_data.
    total = np.sum(data_dist)
    diff = int(num_data - total)
    sign = np.sign(diff)
    for i in range(abs(diff)):
        c = cliques[i % len(cliques)]
        counts[c] += sign

    return counts


a = distribute_data(
    24,
    3,
    3,
    10000)

# %%

NUM_SIMULATIONS = 10
SIM_DURATION = datetime.timedelta(days=365 * 4)

NUM_MACHINES = 100
NUM_PARTITIONS = 20
MEAN_TIME_TO_FAILURE = datetime.timedelta(days=30 * 10)
MEAN_TIME_TO_REPAIR = datetime.timedelta(days=3)
NUM_REPLICAS = 2
NUM_DATA: int = int(1e6)
RECOVERY_TIME = datetime.timedelta(days=7)


def sim_time_to_repair_dist(size: int) -> np.ndarray:
    return time_to_repair_uniform_dist(size, MEAN_TIME_TO_REPAIR)


# %%
failure_starts = time_to_failure_exp_dist(
    NUM_MACHINES,
    mttf=MEAN_TIME_TO_FAILURE,
    sim_duration=SIM_DURATION)

outages = gen_machine_outages(
    num_machines=NUM_MACHINES,
    mttf=MEAN_TIME_TO_FAILURE,
    time_to_repair_dist=sim_time_to_repair_dist,
    sim_duration=SIM_DURATION)

outage_cliques = find_outage_cliques(outages, clique_size=NUM_REPLICAS)

data_count_by_clique = distribute_data(
    NUM_MACHINES, NUM_PARTITIONS, NUM_REPLICAS, NUM_DATA)

full_outages = [oc for oc in outage_cliques if oc.machines in data_count_by_clique]
print(f'Full outages: {len(full_outages)}')
print()
print(OutageClique.table_header())
print('\n'.join(o.format() for o in full_outages))

# %%
