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


def gen_machine_failure_starts(
        num_machines: int,
        failure_dist: RandomDistFn,
        sim_duration: datetime.timedelta) -> np.ndarray:
    """Generates a two-dimensional array of monotonically increasing failure
    times of integer hours from the give failure distribution.

    The returned ndarray has `num_machines` rows.  Each row represents the
    start times of failures for a single machine.  Each row has the same number
    of columns.  The last value of each column is guaranteed to exceed the
    simulation duration.

    For example, for 2 machines this method might produce:

        [[1, 5, 22], [7, 9, 15]]

    Meaning machine 0 has a failures that start at hours 1, 5, and 22.
    Machine 1 has failures that start at hours 7, 9, and 15.

    The exponential distribution is a good candidate for the reasons described
    in 'Availability in Globally Distributed Storage Systems' [1].

    > The exponential distribution is a reasonable approximation for the
    > following reasons. First, the Weibull distribution is a generalization
    > of the exponential distribution that allows the rate parameter to
    > increase over time to reflect the aging of disks. In a large
    > population of disks, the mixture of disks of different ages tends to
    > be stable, and so the average failure rate in a cell tends to be
    > constant. When the failure rate is stable, the Weibull distribution
    > provides the same quality of fit as the exponential. Second, disk
    > failures make up only a small subset of failures that we examined, and
    > model results indicate that overall availability is not particularly
    > sensitive to them. Finally, other authors ([24]) have concluded that
    > correlation and non-homogeneity of the recovery rate and the mean time
    > to a failure event have a much smaller impact on system-wide
    > availability than the size of the event.

    [1]: http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36737.pdf
    """
    sim_hours = total_hours(sim_duration)
    # Use a generous fudge factor so we only have to drop columns, not add
    # new columns to ensure the min(last_column) > sim_duration.  Surely, there's
    # a cleaner way to do this.
    fudge_factor = 10
    # Assume machines last at least 1 month
    hours_per_month = 24 * 30
    num_cols = (sim_hours // hours_per_month) * fudge_factor

    all_failures = failure_dist(num_machines * num_cols)
    reshaped = np.reshape(all_failures, (num_machines, num_cols))
    summed = np.cumsum(reshaped, 1)

    # Drop columns after the first column where min(col) > sim_duration.
    col_mins = np.min(summed, axis=0)
    cols_exceeding_sim_duration = np.nonzero(col_mins > sim_hours)[0]
    assert cols_exceeding_sim_duration.size > 0, (
        'Not enough samples to exceed sim duration for all columns')
    # Add 1 because we want to keep the column we found.
    cols_to_keep = cols_exceeding_sim_duration[0] + 1
    return np.delete(summed, np.s_[cols_to_keep:], axis=1)


def uniform_dist(
        mean: datetime.timedelta,
        stddev: datetime.timedelta,
        size: int) -> np.ndarray:
    """Returns an one-dimensional array of integers drawn from a normal
    distribution.

    The min value is 1.
    """
    mean_hours = total_hours(mean)
    stddev_hours = total_hours(stddev)
    normal_dist = np.random.normal(mean_hours, scale=stddev_hours, size=size)
    return np.clip(np.rint(normal_dist), a_min=1, a_max=None)


def exp_dist(scale: datetime.timedelta, size: int) -> np.ndarray:
    """Returns an one-dimensional array of integers drawn from an exponential
    distribution.

    The min value is 1.
    """
    exp = np.random.exponential(total_hours(scale), size)
    return np.clip(np.rint(exp), a_min=1, a_max=None)


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
        time_to_failure_dist: RandomDistFn,
        time_to_repair_dist: RandomDistFn,
        sim_duration: datetime.timedelta) -> List[Outage]:
    """Generates a list of outages for num_machines.

    The outages for each machine are drawn from an exponential distribution using
    mean time to failure (mttf).  The outage duration is failure start time plus
    the duration in hours drawn from time_to_repair_dist.
    """
    failure_starts = gen_machine_failure_starts(
        num_machines=num_machines,
        failure_dist=time_to_failure_dist,
        sim_duration=sim_duration,
    )
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
    """An outage clique s when multiple machines are down over the same
    time period.

    The start_hour for a clique is max of each MachineOutage.start_hour.
    The end_hour for a clique is the min of each MachineOutage.end_hour.
    In other words, the duration from start_hour to end_hour is guaranteed
    to be fully contained in each individual machine outage that contributes
    to this clique.
    """
    # Order is important, we want to sort by hours first.
    start_hour: int
    end_hour: int
    machines: FrozenSet[int]

    @staticmethod
    def table_header():
        def under(word):
            return '=' * len(word)

        return (f'{"outage start":<25} {"duration":<20} {"machines"}\n'
                f'{under("outage start"):<25} {under("duration"):<20} '
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
    """Finds all outage buddies where exactly clique_size machines were down
    in the same time period.

    The clique size will generally be the number of replicas because that
    potentially indicates that data is completely unavailable.
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
            # The Python for-else statement means the inner loop didn't break,
            # so we found a clique.
            machines = frozenset(x.machine for x in outages[i:i + clique_size])
            assert len(machines) == clique_size, 'Found same machine in clique'
            cliques.append(
                OutageClique(
                    start_hour=clique_start,
                    end_hour=clique_end,
                    machines=machines))
    return cliques


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
    all_outages: List[Outage]
    full_outages: List[OutageClique]


def run_sim(cfg: SimConfig) -> SimResult:
    data_count_by_clique = distribute_data(
        cfg.num_machines, cfg.num_partitions, cfg.num_replicas, cfg.num_data)
    outages = gen_machine_outages(
        num_machines=cfg.num_machines,
        time_to_failure_dist=cfg.time_to_failure_dist,
        time_to_repair_dist=cfg.time_to_repair_dist,
        sim_duration=cfg.sim_duration,
    )
    outage_cliques = find_outage_cliques(outages, clique_size=cfg.num_replicas)
    full_outages = [
        oc for oc in outage_cliques if oc.machines in data_count_by_clique
    ]

    return SimResult(
        config=cfg,
        all_outages=outages,
        full_outages=full_outages,
    )


# %%
BASE_CONFIG = SimConfig(
    num_machines=70,
    sim_duration=years(70),
    num_partitions=14,
    num_replicas=2,
    num_data=int(1e6),
    time_to_failure_dist=lambda size: exp_dist(years(4), size),
    time_to_repair_dist=lambda size: uniform_dist(days(3), hours(6), size)
)


# %%
# See what happens if we vary the time to failure.
def sim_outages_per_mttf(
        num_sims: int,
        base_config: SimConfig) -> Dict[int, List[float]]:
    outages_by_mttf = collections.defaultdict(list)
    bi_annual_months = range(6, 5 * 12 + 1, 6)
    for month in bi_annual_months:
        cfg = dataclasses.replace(
            base_config,
            time_to_failure_dist=lambda size: exp_dist(months(month), size))
        for _ in range(num_sims):
            # If there's a lot of variance, there might not be any full outages
            # so use the sim duration instead.
            first_failure = total_hours(cfg.sim_duration)
            result = run_sim(cfg)
            if result.full_outages:
                first_failure = result.full_outages[0].start_hour

            outages_by_mttf[month].append(first_failure / (24 * 365))
    return outages_by_mttf


outages_per_mttf = sim_outages_per_mttf(
    num_sims=100, base_config=BASE_CONFIG)

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
def sim_outages_per_num_partitions(
        num_sims,
        base_config: SimConfig) -> Dict[int, List[float]]:
    """See what happens if we vary the number of partitions."""
    # Create an abundant number (one that has many divisors) so there's more
    # partitions that evenly divide the number of partitions.
    num_machines = 60

    outages_by_partition = collections.defaultdict(list)
    for partition in [1, 2, 3, 4, 5, 10, 12, 15, 20, 30]:
        cfg = dataclasses.replace(
            base_config, num_machines=num_machines, num_partitions=partition)
        for _ in range(num_sims):
            # If there's a lot of variance, there might not be any full outages
            # so default to the sim duration.
            first_failure = total_hours(cfg.sim_duration)
            result = run_sim(cfg)
            if result.full_outages:
                first_failure = result.full_outages[0].start_hour

            outages_by_partition[partition].append(first_failure / (24 * 365))

    return outages_by_partition


outages_per_partition = sim_outages_per_num_partitions(
    num_sims=100, base_config=BASE_CONFIG)

# %%
plt.interactive(False)
pd.DataFrame(outages_per_partition).boxplot()
plt.pyplot.title('Time to data unavailability by number of partitions')
plt.pyplot.xlabel('Number of distinct partitions')
plt.pyplot.ylabel('Time to data unavailability in years')
plt.pyplot.savefig('outages-by-partition.png', dpi=300)
plt.pyplot.show()
