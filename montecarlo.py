"""
A monte-carlo simulation to model failures in a distributed storage system.

Math StackExchange question: https://math.stackexchange.com/q/3217875/31502

In broad strokes, the simulation proceeds as follows:

-   Partition NUM_MACHINES into NUM_PARTITIONS separate groups.  This simulation
    uses partition to refer to separate groups instead of `cell` used by the math
    StackExchange question.

-   Uniformly distribute NUM_DATA pieces of data to all subsets of machines such
    that:

    1.  Every subset of machines reside in the same partition.
    2.  Every subset has exactly a size of NUM_REPLICAS.

-   Generate machine failure start times pulling samples from an exponential
    distribution.

-   Get the cumulative sum of the failure times to generate subsequent failure
    times for a machine.  Meaning, turn [1, 3, 2, 7] into [1, 4, 6, 13].

-   Create an outage for a machine by adding the time to repair to the failure
    start time.  The time to repair is drawn from a normal distribution.

-   Find all outages where N machines are down at the same time.  This is an
    outage clique.  When N == NUM_REPLICAS, this means we might have an outage
    for some subset of data.

-   Find all outage cliques where each machine in the clique hosts the same
    piece of data.  The found cliques mean some data is completely unavailable.

-   Display the outage cliques start time, duration, and machines in the clique.

"""
# %% Imports
import bisect
import collections
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, FrozenSet, NewType, Callable
import itertools

import datetime
import numpy as np
import pandas as pd
import matplotlib as plt
import humanize

# %%

# A random distribution that takes a int size and returns samples
# from the distribution.  For example, here's a random dist from
# a uniform distribution with a mean of 5 and stddev of 2:
#
#     lambda size: np.random.normal(5, 2, size)
RandomDistFn = NewType('RandomDist', Callable[[int], np.ndarray])


def hours(n: int) -> datetime.timedelta:
    """Creates a timedelta of exactly n hours."""
    return datetime.timedelta(hours=n)


def days(n: int) -> datetime.timedelta:
    """Creates a timedelta of exactly n days."""
    return datetime.timedelta(days=n)


def months(n: int) -> datetime.timedelta:
    """Creates a timedelta of n months where a month is defined as
    30.5 days."""
    return days(int(n * 30.5))


def years(n: int) -> datetime.timedelta:
    """Creates a timedelta of n years where a year is defined as
    365 days."""
    return days(n * 365)


def total_hours(delta: datetime.timedelta) -> int:
    """Converts a timedelta into the number of hours.

    Timedelta normalizes the delta into days, seconds and microseconds so
    'delta.hours' only returns the hours passed into the constructor.  We
    typically want the total elapsed time."""
    return int(delta.total_seconds()) // 3600


def total_days(delta: datetime.timedelta) -> int:
    """Converts a timedelta into the total number of days.

    See total_hours."""
    return total_hours(delta) // 24


def total_months(delta: datetime.timedelta) -> int:
    """Converts a timedelta into the total number of months.

    See total_hours."""
    return int(total_days(delta) / 30.5)


def normal_dist(
        mean: datetime.timedelta,
        stddev: datetime.timedelta,
        size: int) -> np.ndarray:
    """Returns an one-dimensional array of integers drawn from a normal
    distribution.

    The min value is 1.
    """
    mean_hours = total_hours(mean)
    stddev_hours = total_hours(stddev)
    dist = np.random.normal(mean_hours, scale=stddev_hours, size=size)
    return np.clip(np.rint(dist), a_min=1, a_max=None)


def exp_dist(scale: datetime.timedelta, size: int) -> np.ndarray:
    """Returns an one-dimensional array of integers drawn from an exponential
    distribution.

    The min value is 1.
    """
    exp = np.random.exponential(total_hours(scale), size)
    return np.clip(np.rint(exp), a_min=1, a_max=None)


# %%
def gen_machine_failure_starts(
        num_iterations: int,
        num_machines: int,
        time_to_failure_dist: RandomDistFn,
        sim_duration: datetime.timedelta) -> np.ndarray:
    """Generates a three-dimensional array of monotonically increasing failure
    times of integer hours from the give failure distribution.

    The returned ndarray has shape (num_iterations, num_machines, COLS) rows.
    Each row in an iteration represents the start times of failures for a single
    machine.  Each row has the same number of columns.  The last value of each
    column is guaranteed to be less than the simulation duration.

    For example, for 2 machines this method might produce:

        [[1, 5, 22], [7, 9, 15]]

    Meaning machine 0 has a failures that start at hours 1, 5, and 22.
    Machine 1 has failures that start at hours 7, 9, and 15.

    The failure times are drawn from the time_to_failure_dist.
    """
    min_hours = total_hours(sim_duration)
    num_rows = num_iterations * num_machines
    # Get the 30th percentile as a reasonable guess for how many samples we need.
    # Use a lower percentile to increase num_cols and avoid looping in most cases.
    p30_val = np.quantile(time_to_failure_dist(20), 0.3)
    # Generate at least 10 columns each time.
    num_cols = max(int(min_hours / p30_val), 10)
    storage = []

    while True:
        starts = time_to_failure_dist(size=(num_rows, num_cols)).cumsum(axis=1)
        is_larger = starts[:, -1] >= min_hours
        good_rows = starts[is_larger, :]
        storage.append(good_rows)

        number_of_good_rows = sum([_a.shape[0] for _a in storage])
        if number_of_good_rows >= num_machines:
            starts = np.vstack(storage)
            break

    # Only keep columns before the column where each value >= min_hours.
    cols_to_keep = np.logical_not(np.all(starts >= min_hours, axis=0))
    min_num_cols = np.nonzero(cols_to_keep)[0].size
    return starts[:, cols_to_keep].reshape(
        num_iterations, num_machines, min_num_cols)


# %%
@dataclass(frozen=True, order=True)
class Outage:
    """An outage is the period of time where a machine is down."""
    # Order is important, we want to sort by hours first to ease finding
    # overlaps in outages.
    start_hour: int
    end_hour: int
    machine: int


def find_overlap_masks(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Returns a boolean mask where the start of any overlap is True.

    This only finds adjacent overlaps and will miss cases where a cell
    overlaps another cell that's not adjacent like:

        [[0, 10], [4, 6], [8, 11]]

        # Mask
        [True, False, False]

    Even though [0, 10] overlaps everything, this function only detects the
    first overlap.
    """
    # Shift starts left one.
    starts_shift = np.roll(starts, -1, axis=2)
    starts_shift[:, :, -1] = int(2 ** 62)

    # An overlap is anywhere that end - starts_shift is greater than or
    # equal to 0.
    return (ends - starts_shift) >= 0


def machine_ids(num_iterations, num_machines, num_cols):
    ids = np.transpose(
        np.arange(num_machines)
            .repeat(num_cols * num_iterations)
            .reshape(num_machines, num_cols, num_iterations),
        axes=(2, 0, 1))
    return ids


def gen_machine_outages(
        num_iterations: int,
        num_machines: int,
        time_to_failure_dist: RandomDistFn,
        time_to_repair_dist: RandomDistFn,
        sim_duration: RandomDistFn) -> List[np.ndarray]:
    """Generates a list of outages for num_machines run num_iterations times.

    An outage in a 3 element ndarray of [start_time, end_time, machine_id].  The
    length of the returned list is num_iterations.

    The outages start times for each machine are drawn from the
    time_to_failure_dist.  The outage end times is the start time plus the
    repair times drawn from the time_to_repair_dist.
    """
    starts = gen_machine_failure_starts(
        num_iterations=num_iterations,
        num_machines=num_machines,
        time_to_failure_dist=time_to_failure_dist,
        sim_duration=sim_duration,
    )
    repairs = time_to_repair_dist(starts.size).reshape(starts.shape)
    ends = starts + repairs
    ends = np.clip(ends, a_min=None, a_max=total_hours(sim_duration), out=ends)
    num_cols = starts.shape[-1]
    overlap_mask = find_overlap_masks(starts, ends)

    coalesced_indexes = []
    overlap_start_indexes = np.transpose(np.nonzero(overlap_mask))
    for i, j, k in overlap_start_indexes:
        cur_end = ends[i, j, k]

        # Coalesce by setting the cur_end to largest end in any of the
        # following ends that overlap cur_end.
        for k2 in range(k + 1, ends[i, j].size):
            next_start = starts[i, j, k2]
            next_end = ends[i, j, k2]
            if cur_end < next_start:
                break
            cur_end = max(cur_end, next_end)
            ends[i, j, k] = cur_end
            # Translate from 3d to 1d.
            coalesced_indexes.append(i * num_machines * num_cols + j * num_cols + k2)

    ids = machine_ids(num_iterations, num_machines, num_cols)
    stacked = np.stack((starts, ends, ids), axis=-1)
    flattened = stacked.reshape(starts.size, stacked.shape[-1])
    # Drop outages where the start is after the sim_duration.
    rows_to_keep = flattened[:, 0] < total_hours(sim_duration)
    # Drop outages that we already coalesced so they're not double counted.
    rows_to_keep[coalesced_indexes] = False

    # Split into a python list so the arrays can be different sizes.
    splits = [x.reshape(num_machines * num_cols, stacked.shape[-1])
              for x in np.split(flattened, num_iterations)]
    split_masks = np.split(rows_to_keep, num_iterations)

    # Apply the masks.
    for i in range(num_iterations):
        mask = split_masks[i]
        masked = splits[i][mask]
        splits[i] = masked

    return splits


seedSet = 1558056446
np.random.seed(seedSet)
a = gen_machine_outages(
    num_iterations=2,
    num_machines=2,
    time_to_failure_dist=lambda size: normal_dist(hours(1), hours(6), size),
    time_to_repair_dist=lambda size: exp_dist(hours(3), size),
    sim_duration=hours(20))
print('orig', a)


def find_outage_cliques(
        all_outages: List[np.ndarray],
        num_machines: int,
        num_partitions: int,
        num_replicas: int) -> List[np.ndarray]:
    """Finds outage cliques when multiple machines are down over the same
    time period that are in the same partition.

    Each element in the return list is all the outage cliques for a single
    iteration.  Each element in the ndarray is:

        [start_hour, end_hour, machine_1, machine_2]

    Only a clique size of 2 (corresponding to num_machines // num_partitions)
    is supported.
    """
    machines_per_part = num_machines // num_partitions
    return [find_outage_cliques_single(o, machines_per_partition=machines_per_part, clique_size=num_replicas)
            for o in all_outages]


def find_outage_cliques_single(
        outages: np.ndarray,
        machines_per_partition: int,
        clique_size: int) -> np.ndarray:
    assert clique_size <= 2, 'Larger clique sizes than 2 unsupported'
    # Sort by start and end column. Requires structured arrays.
    # https://stackoverflow.com/questions/2828059
    view = outages.view(dtype=[('start', 'f8'), ('end', 'f8'), ('id', 'f8')])
    view.sort(order=['start', 'end'], axis=0)

    starts = outages[:, 0]
    ends = outages[:, 1]
    machines = outages[:, 2]

    starts_shift = np.roll(starts, -1, axis=0)
    # Use a large value so it appears as False in the mask.
    starts_shift[-1] = ends[-1]
    ends_shift = np.roll(ends, -1, axis=0)
    machines_shift = np.roll(machines, -1, axis=0)

    mask = ends - starts_shift > 0
    # Mask out machines that aren't in the same partition.
    m1 = machines // machines_per_partition
    m2 = machines_shift // machines_per_partition
    same_partition_mask = m1 == m2
    mask &= same_partition_mask

    outage_starts = starts_shift[mask]
    outage_ends = np.min(np.dstack((ends, ends_shift)), axis=2)[0][mask]
    outage_m1 = machines[mask]
    outage_m2 = machines_shift[mask]
    return np.dstack((outage_starts, outage_ends, outage_m1, outage_m2))[0]


find_outage_cliques_single(a[0], clique_size=2, machines_per_partition=2)


# %%

# %%


# %%
@dataclass(frozen=True)
class MachinePartition:
    """A partition is a group of machines such that each piece of data is
    replicated in exactly once partition.

    Each partition contains the same number of machines.  Typically, the size
    of each partition is the num_total_machines / num_partitions.
    """
    machines: FrozenSet[int]


def gen_partitions(num_machines: int, num_partitions: int) -> List[MachinePartition]:
    """Evenly divides num_machines into num_partitions."""
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
    """Uniformly distributes num_data pieces of data in num_partitions.

    - Data is replicated exactly num_replicas times.
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

    # Add or subtract data so it equals num_data. We need this because we did
    # the probabilistic distribution above.
    total = np.sum(data_dist)
    diff = int(num_data - total)
    sign = np.sign(diff)
    for i in range(abs(diff)):
        c = cliques[i % len(cliques)]
        counts[c] += sign

    return counts


# %%
@dataclass(frozen=True)
class SimConfig:
    """Configuration info for how to run a simulation."""
    num_iterations: int
    num_machines: int
    sim_duration: datetime.timedelta
    num_partitions: int
    num_replicas: int
    num_data: int
    time_to_failure_dist: RandomDistFn
    time_to_repair_dist: RandomDistFn


@dataclass(frozen=True)
class SimResult:
    """Information about outages for a simulation."""
    config: SimConfig
    full_outages: List[np.ndarray]


def run_sim(cfg: SimConfig) -> SimResult:
    """Runs the simulation according to params in cfg.

    The returned list is exactly SimConfig.num_iterations long."""
    all_outages = gen_machine_outages(
        num_iterations=cfg.num_iterations,
        num_machines=cfg.num_machines,
        time_to_failure_dist=cfg.time_to_failure_dist,
        time_to_repair_dist=cfg.time_to_repair_dist,
        sim_duration=cfg.sim_duration)
    all_cliques = find_outage_cliques(
        all_outages,
        num_machines=cfg.num_machines,
        num_partitions=cfg.num_partitions,
        num_replicas=cfg.num_replicas)

    return SimResult(config=cfg, full_outages=all_cliques)


# %%
BASE_CONFIG = SimConfig(
    num_iterations=100,
    num_machines=70,
    sim_duration=years(70),
    num_partitions=14,
    num_replicas=2,
    num_data=int(1e6),
    time_to_failure_dist=lambda size: exp_dist(years(4), size),
    time_to_repair_dist=lambda size: normal_dist(days(3), hours(6), size)
)


# %%
def sim_outages_per_mttf(base_cfg: SimConfig) -> Dict[int, List[float]]:
    """See what happens if we vary the time to failure."""
    outages_by_mttf = collections.defaultdict(list)

    bi_annual_months = range(6, 5 * 12 + 1, 6)
    for month in bi_annual_months:
        cfg = dataclasses.replace(
            base_cfg,
            time_to_failure_dist=lambda size: exp_dist(months(month), size))

        result = run_sim(cfg)
        for outages in result.full_outages:
            # If there's a lot of variance, there might not be any full outages
            # so default the sim duration instead.
            first_fail = total_hours(cfg.sim_duration)
            if outages.size > 0:
                first_fail = outages[0, 0]
            outages_by_mttf[month].append(first_fail / (24 * 365))

    return outages_by_mttf


# %%
outages_per_mttf = sim_outages_per_mttf(BASE_CONFIG)

# Hard to see y-axis with all months so truncate data a bit.
outages_per_mttf_30_months = {k: v for k, v in outages_per_mttf.items() if k <= 30}


# %%
plt.interactive(False)
pd.DataFrame(outages_per_mttf).boxplot()
plt.pyplot.title('Time to data unavailability by machine MTTF')
plt.pyplot.xlabel('Machine mean time to failure in months with an exponential distribution')
plt.pyplot.ylabel('Time to data unavailability in years')
plt.pyplot.savefig('outages-by-mttf-60-months.png', dpi=300)
plt.pyplot.show()

pd.DataFrame(outages_per_mttf_30_months).boxplot()
plt.pyplot.title('Time to data unavailability by machine MTTF')
plt.pyplot.xlabel('Machine mean time to failure in months with an exponential distribution')
plt.pyplot.ylabel('Time to data unavailability in years')
plt.pyplot.savefig('outages-by-mttf-30-months.png', dpi=300)
plt.pyplot.show()


# %%
def sim_outages_per_num_partitions(base_cfg: SimConfig) -> Dict[int, List[float]]:
    """See what happens if we vary the time to failure."""
    outages_by_partition = collections.defaultdict(list)

    # Create an abundant number (one that has many divisors) so there's more
    # partitions that evenly divide the number of partitions.
    num_machines = 60

    for partition in [1, 2, 3, 4, 5, 10, 12, 15, 20, 30]:
        cfg = dataclasses.replace(
            base_cfg,
            num_machines=num_machines,
            num_partitions=partition)

        result = run_sim(cfg)
        for outages in result.full_outages:
            # If there's a lot of variance, there might not be any full outages
            # so default the sim duration instead.
            first_fail = total_hours(cfg.sim_duration)
            if outages.size > 0:
                first_fail = outages[0, 0]
            outages_by_partition[partition].append(first_fail / (24 * 365))

    return outages_by_partition


# %%
outages_per_partition = sim_outages_per_num_partitions(base_cfg=BASE_CONFIG)

# # %%
# plt.interactive(False)
# pd.DataFrame(outages_per_partition).boxplot()
# plt.pyplot.title('Time to data unavailability by number of partitions')
# plt.pyplot.xlabel('Number of distinct partitions')
# plt.pyplot.ylabel('Time to data unavailability in years')
# plt.pyplot.savefig('outages-by-partition.png', dpi=300)
# plt.pyplot.show()

# %%
failure_starts = gen_machine_failure_starts(
    num_machines=10,
    failure_dist=lambda size: exp_dist(hours(8), size),
    sim_duration=days(1),
)
# repair_times = time_to_repair_dist(failure_starts.size).reshape(failure_starts.shape)
# failure_ends = failure_starts + repair_times
# start_ends = np.dstack((failure_starts, failure_ends))
