# Temporal encounter graph audit

## Graph definition

- Node: one eligible hospital encounter.
- Edge: `source_current -> target_previous`, linking an encounter to the immediately preceding eligible encounter from the same patient.
- Graphs: train, validation, and test are constructed separately; no cross-split edges exist.
- Labels: stored separately from structural node files and never used to create edges.
- Ordering: numeric `encounter_id` is used as a chronological surrogate within patient. UCI documents it as a unique encounter identifier, not a timestamp, so exact chronology and time gaps remain a limitation.

## Audit

| Split | Encounter nodes | Causal history edges | Patients | Repeated patients | Nodes with history | Max encounters/patient |
|---|---:|---:|---:|---:|---:|---:|
| train | 69,721 | 20,628 | 49,093 | 11,430 | 20,628 (29.59%) | 40 |
| validation | 14,683 | 4,238 | 10,445 | 2,421 | 4,238 (28.86%) | 20 |
| test | 14,939 | 4,487 | 10,452 | 2,490 | 4,487 (30.04%) | 28 |

Total: **99,343 encounter nodes** and **29,353 directed history edges**.

## Leakage controls

1. A current encounter points backward to its previous encounter; future encounters are not sampled as neighbours.
2. Any message-passing implementation must preserve direction and aggregate only from the current node toward prior nodes. Symmetrizing these edges would leak future information.
3. Patient assignment occurred before graph construction, so one patient's encounters cannot bridge train, validation, and test.
4. Node labels are stored in separate local files. Edge existence never depends on readmission labels.
5. Patient identifiers are replaced by salted local pseudonymous tokens, and all graph artifacts remain Git-ignored.

## Interpretation

Only encounters with an observed earlier encounter can benefit directly from temporal edges. First observed encounters remain isolated with respect to longitudinal history and must rely on their tabular features. This makes the graph comparison clinically interpretable: any improvement should be concentrated among patients with repeated observed encounters.
