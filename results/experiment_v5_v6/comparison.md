# ApacheJIT comparison

Seed-level metrics precede averaging; AP is average precision.

| Method | MCC mean | MCC sample SD | AP mean |
|---|---:|---:|---:|
| cc2vec | 0.1229 | 0.0162 | 0.2537 |
| deepjit | 0.3453 | 0.0089 | 0.4313 |
| jitline_apachejit | 0.3596 | 0.0097 | 0.4610 |

See comparison.json for per-project scores, stress coverage, matched cohorts, confidence intervals and corrected tests.

Intervals are conditional on this benchmark; related projects and commits limit independence.
