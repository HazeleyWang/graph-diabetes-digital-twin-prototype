# Temporal graph design

## Objective

Represent observed longitudinal hospital history without exposing future encounters to the prediction for a current encounter.

## Minimal prototype

```text
current encounter ──previous_encounter──> earlier encounter ──> still earlier encounter
```

Each encounter is a node with its own admission, diagnosis, laboratory, and medication features. A directed edge points backward to the immediately preceding eligible encounter for that patient. Multiple message-passing steps can therefore summarize deeper history.

## Why not use a patient hub yet?

A patient node connected simultaneously to all encounters could aggregate a future encounter and send it back to an earlier encounter. Preventing this requires time-sliced patient states or strictly causal sampling. The first prototype avoids that shortcut by using backward-only encounter chains.

## Why not build patient-similarity edges yet?

Similarity edges require additional choices about scaling, categorical distance, nearest neighbours, and whether test-patient information can influence training representations. They are postponed until the temporal graph baseline is verified.

## Limitation

The dataset has encounter identifiers but no explicit admission timestamps. Numeric encounter order is therefore treated as a surrogate ordering, not as verified elapsed time. We will not model time gaps or claim a continuous-time digital twin from this dataset.

