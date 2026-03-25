# epoxy_HPC

## Model notes

The `kapp` model (`BatchBayesian_simple.py`) — which uses `kapp * a^m * (1-a)^(n/2) * (r-a)^(n/2)` — was tested but abandoned. Dropping the k1 term means the reaction rate goes to zero as α→0, which produces poor fits and very wide posterior CIs for datasets that plateau early (e.g. NMR at low temperature). The full Kamal-Malkin form `(k1 + k2*a^m) * (1-a)^(n/2) * (r-a)^(n/2)` is used instead.
