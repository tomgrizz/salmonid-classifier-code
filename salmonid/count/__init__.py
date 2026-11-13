"""count sub-package: from fish tracks to daily counts, and the counting rule.

A track becomes a count by three decisions: its species, the day it passed, and whether it
moved upstream, downstream or neither (the 50 px / 300 px rule; ``docs/COUNTING_METHOD.md``).

Published path, in the order they are run:

    run_corpus                  runs the whole chain for the seasons named, step by step
    corpus_countr_from_store    one season's tracks -> daily model and technician counts
                                (Tier 2); also writes the files in data/net_rule_sweep/
    net_rule_loyo               re-chooses the counting rule with each season left out in
                                turn (Tier 1; manuscript Section 3.4)

Earlier analyses, kept because they led to the counting rule or to checks made during
development (see each module's docstring for what it used and what it showed):

    net_threshold_sweep, net_direction_decomp, track_reaggregate
    corpus_fullyear_vs_tech, corpus_2024_vs_tech, corpus_historical_vs_tech,
    corpus_multiyear_vs_tech
"""
