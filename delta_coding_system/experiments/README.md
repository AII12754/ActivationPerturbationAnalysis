# Experiments Archive

This directory stores historical exploration code and reports that led to the current production default strategy.

Contents:
- compression_experiment_*.py: exploratory evaluation scripts, ablations, and phasewise studies
- reports/: generated writeups and design notes from those studies

Status:
- These files are kept for traceability and reproduction.
- The production system no longer depends on them for default runtime behavior.
- New development should start from the core modules in delta_coding_system/.
