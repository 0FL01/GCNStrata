# Single lower-live-state revision

This isolated bundle uses in-place low-byte key normalization in the existing count scan after a singleton third-byte stop. It removes the interval upper bound, independent recovery flag, mutable recovered threshold, and shared full-key writer/readback. The source candidate is default-off. The original baseline, flags, headers, wrappers, resource checker, and resource-checker negative tests are unchanged. No device compilation or GPU launch has been run for this bundle by its preparer.

The resource checker is intentionally byte-identical to v1. Its old informational manual-review sentence mentions a single writer and full-key recovery; these are absent by design in v2. Manual review must instead verify uniform stop, correct low-byte mask, in-place normalization, unchanged ascending output, guards, and barriers. This wording does not change any resource or ISA acceptance criterion. A failed gate remains a hard stop.

The stopped numeric threshold is a 24-bit prefix, not the original complete threshold. Every positive block has the same greater/equal/less classification after normalization as against the exact full threshold. Exact selected IDs and lowest-index tie resolution remain the contract. CPU tests model block synchronization and scans; they are not hardware qualification.
