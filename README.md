# Distributed storage monte-carlo simuation

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

## Running the simulation:

```bash
git clone https://github.com/jschaf/cellarch.git
cd cellarch

# Create a venv dir in the cellarch dir.
python3 -m venv venv
source venv/bin/activate

# Verify the venv python is used.
which python

pip install -r requirements.txt 

# Run it
python montecarlo.py
```
