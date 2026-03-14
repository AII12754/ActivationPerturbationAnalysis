"""Legacy src package — deprecated in favor of activation_science.

All modules in this package continue to work but are superseded by the
unified ``activation_science`` package.  Migration guide:

    Old import                              New import
    ─────────────────────────────────────── ──────────────────────────────────────────
    src.model                               activation_science.core.model
    src.prompts                             activation_science.core.datasets
    src.scheduler                           activation_science.core.scheduler
    src.metrics                             activation_science.metrics.similarity
    src.decode_experiment                    activation_science.experiments.e0_decode_similarity
    src.decode_storage                      activation_science.core.storage
    src.decode_sweep                        activation_science.runners.run_experiment (e0)
    src.decode_analysis                     activation_science.analysis.per_experiment.e0_plots
    src.residual_experiment                 activation_science.experiments.e1_residual_decomposition
    src.logit_lens_experiment               activation_science.experiments.e2_logit_lens
    src.geometry_experiment                 activation_science.experiments.e3_representation_geometry
    src.cross_sequence_experiment           activation_science.experiments.e4_cross_sequence_alignment
    src.state_experiment                    activation_science.experiments.e5_state_detection

Run scripts (deprecated):
    python run_decode_sweep.py        →  python -m activation_science.runners.run_experiment e0
    python run_residual_sweep.py      →  python -m activation_science.runners.run_experiment e1
    python run_decode_analysis.py     →  python -m activation_science.runners.run_analysis e0
"""
