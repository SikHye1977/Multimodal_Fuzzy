"""Shared data/runtime API. Importing this package does not run experiments."""
from .data import (SPLITS, MODALITIES, EF, QF, EP, QP, USER, MANIFESTS,
                   VirtualUser, Embeddings, make_users, make_pairs, finger_labels,
                   negative_permutation, training_batches, model_inputs, forward_pair)
from .runtime import (load_model_module, sha256, write_json, write_csv, checkpoint,
                      duration, Progress, Log)
from .paths import ROOT
