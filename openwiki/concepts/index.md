# Files

- [Disk cache](disk-cache.md) - The persistent pickle and JSON-index cache for dataset downloads and builder results. It reuses a containing download cropped with Core's own footprint rule, keys builders by metadata fingerprints, evicts by TTL and size, and is shared across Sessions today.
- [Dispatch, object references and serialization](dispatch-and-object-store.md) - How run_operation resolves parameters, calls a dtcc-core function or dataset, stores the result in the calling Session's ObjectStore under a short ID, and returns an LLM-sized summary, plus the object tools built on that store.
- [Operation catalogue](operation-catalogue.md) - How registry.py reflects over the pinned dtcc-core into a catalogue of named Operations with parameter schemas, built once per process (HTTP at startup, stdio on first use), failing loudly on a broken Core, with dtcc-sim datasets joining later from a background retrier.
