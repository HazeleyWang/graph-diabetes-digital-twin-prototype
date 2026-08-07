"""Build leakage-aware temporal encounter graphs for each patient split."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "temporal_graph"
REPORT_PATH = PROJECT_ROOT / "reports" / "temporal_graph_audit.md"


def validate_temporal_edges(nodes: pd.DataFrame, labels: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Enforce graph invariants before any row-level artifact is written."""
    node_ids = set(nodes["node_id"])
    assert edges["source_current"].isin(node_ids).all()
    assert edges["target_previous"].isin(node_ids).all()
    assert (
        pd.to_numeric(edges["source_current"])
        > pd.to_numeric(edges["target_previous"])
    ).all()
    assert nodes["node_id"].is_unique
    assert labels["node_id"].is_unique


def patient_token(patient_nbr: str) -> str:
    """Create a local pseudonymous grouping token; never commit the mapping."""
    value = f"graph-patient-v1:{patient_nbr}".encode()
    return hashlib.sha256(value).hexdigest()[:16]


def main() -> None:
    encounters = pd.read_csv(
        DATA_PATH,
        usecols=["encounter_id", "patient_nbr", "readmitted"],
        dtype=str,
        keep_default_na=False,
    )
    assignments = pd.read_csv(
        SPLIT_PATH,
        dtype={"encounter_id": str, "patient_nbr": str},
    )
    data = encounters.merge(
        assignments,
        on=["encounter_id", "patient_nbr"],
        validate="one_to_one",
    )
    data = data[data["eligible_primary"]].copy()
    data["encounter_order"] = pd.to_numeric(data["encounter_id"], errors="raise")
    data["patient_token"] = data["patient_nbr"].map(patient_token)
    data = data.sort_values(["patient_nbr", "encounter_order"], kind="stable")
    data["sequence_index"] = data.groupby("patient_nbr").cumcount()
    data["previous_encounter_id"] = data.groupby("patient_nbr")["encounter_id"].shift(1)
    data["target_30d"] = (data["readmitted"] == "<30").astype(int)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for split in ["train", "validation", "test"]:
        subset = data[data["split"] == split].copy()
        nodes = subset[["encounter_id", "patient_token", "sequence_index"]].rename(
            columns={"encounter_id": "node_id"}
        )
        labels = subset[["encounter_id", "target_30d"]].rename(
            columns={"encounter_id": "node_id"}
        )
        edges = subset[subset["previous_encounter_id"].notna()][
            ["encounter_id", "previous_encounter_id"]
        ].rename(
            columns={
                "encounter_id": "source_current",
                "previous_encounter_id": "target_previous",
            }
        )
        edges["relation"] = "previous_encounter"

        validate_temporal_edges(nodes, labels, edges)

        nodes.to_csv(GRAPH_DIR / f"{split}_nodes.csv", index=False)
        edges.to_csv(GRAPH_DIR / f"{split}_edges.csv", index=False)
        labels.to_csv(GRAPH_DIR / f"{split}_labels.csv", index=False)

        patient_sizes = subset.groupby("patient_nbr").size()
        report_rows.append(
            {
                "split": split,
                "nodes": len(nodes),
                "edges": len(edges),
                "patients": subset["patient_nbr"].nunique(),
                "patients_repeated": int((patient_sizes > 1).sum()),
                "nodes_with_history": int((subset["sequence_index"] > 0).sum()),
                "max_sequence": int(patient_sizes.max()),
            }
        )

    summary = pd.DataFrame(report_rows)
    table = "\n".join(
        f"| {row.split} | {row.nodes:,} | {row.edges:,} | {row.patients:,} | "
        f"{row.patients_repeated:,} | {row.nodes_with_history:,} "
        f"({100 * row.nodes_with_history / row.nodes:.2f}%) | {row.max_sequence:,} |"
        for row in summary.itertuples()
    )
    total_nodes = int(summary["nodes"].sum())
    total_edges = int(summary["edges"].sum())
    report = f"""# Temporal encounter graph audit

## Graph definition

- Node: one eligible hospital encounter.
- Edge: `source_current -> target_previous`, linking an encounter to the immediately preceding eligible encounter from the same patient.
- Graphs: train, validation, and test are constructed separately; no cross-split edges exist.
- Labels: stored separately from structural node files and never used to create edges.
- Ordering: numeric `encounter_id` is used as a chronological surrogate within patient. UCI documents it as a unique encounter identifier, not a timestamp, so exact chronology and time gaps remain a limitation.

## Audit

| Split | Encounter nodes | Causal history edges | Patients | Repeated patients | Nodes with history | Max encounters/patient |
|---|---:|---:|---:|---:|---:|---:|
{table}

Total: **{total_nodes:,} encounter nodes** and **{total_edges:,} directed history edges**.

## Leakage controls

1. A current encounter points backward to its previous encounter; future encounters are not sampled as neighbours.
2. Any message-passing implementation must preserve direction and aggregate only from the current node toward prior nodes. Symmetrizing these edges would leak future information.
3. Patient assignment occurred before graph construction, so one patient's encounters cannot bridge train, validation, and test.
4. Node labels are stored in separate local files. Edge existence never depends on readmission labels.
5. Patient identifiers are replaced by salted local pseudonymous tokens, and all graph artifacts remain Git-ignored.

## Interpretation

Only encounters with an observed earlier encounter can benefit directly from temporal edges. First observed encounters remain isolated with respect to longitudinal history and must rely on their tabular features. This makes the graph comparison clinically interpretable: any improvement should be concentrated among patients with repeated observed encounters.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote local graph artifacts to {GRAPH_DIR}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
