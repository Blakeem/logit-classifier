"""Readout backends, one per host that can run a forward pass.

Importing this package pulls in no host. `hf` needs the `[hf]` extra, so it is
imported by name at the point of use. `base` is the single path to the port.
"""
