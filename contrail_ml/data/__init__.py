"""contrail_ml.data — data loaders, collocation, and the training-table builder.

Real loaders (iagos/era5/gfs) need credentials/network and are NOT exercised in
CI. The hermetic test fixture is `synthetic_fallback`, which is loudly guarded
so it can never be mistaken for real observations.
"""
