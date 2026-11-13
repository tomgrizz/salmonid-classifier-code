"""train sub-package.

The launcher that trains the published model (v2.7): checks the training pictures, runs the
held-out leak gate, writes the run receipt, then starts the external DEIMv2 trainer with the
pinned seed. See ``salmonid.train.train_v27``.
"""
