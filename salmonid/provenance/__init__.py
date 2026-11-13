"""provenance sub-package: the record of what a training run used.

    record_run_manifest    writes, before training starts, a "receipt" naming the code version,
                           the settings file, a fingerprint of the training picture list, the
                           random seed and the software versions

The receipt of the published model (v2.7) is kept in the ``provenance/`` folder at the top of
the repository.
"""
