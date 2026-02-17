# -*- coding: utf-8 -*-
"""Snapshot helpers for CAD recognition pipeline."""


def save_cad_snapshot_stages(snapshot, raw, classified, topology, model_inputs):
    snapshot.save_json("10_cad_raw_entities.json", raw)
    snapshot.save_json("11_cad_classified.json", classified)
    snapshot.save_json("12_wall_pairs.json", {
        "inner_bounds": {
            "sw": topology.get("inner_sw_cm"),
            "se": topology.get("inner_se_cm"),
            "ne": topology.get("inner_ne_cm"),
            "nw": topology.get("inner_nw_cm"),
        },
        "debug": topology.get("debug", {}),
    })
    snapshot.save_json("13_opening_assignment.json", {
        "measurements_cm": topology.get("measurements_cm", {}),
        "debug": topology.get("debug", {}),
    })
    snapshot.save_json("14_model_inputs.json", model_inputs or topology.get("measurements_cm", {}))
