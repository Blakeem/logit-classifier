"""Tools built on the classifier core that follow its measured best practice.

A caller gets good results from these tools without working the details out from the base API.
The tools use the core's public API plus four same-wheel host helpers, `_determinism`, `_clip_parts`,
`_resident` and `_transformer`, which stay private because they hold host internals. The tools never
mix into the core, so the core never imports this package.
"""
