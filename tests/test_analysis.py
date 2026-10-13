"""Full analysis path on a simulated panel written to disk in the same format as real features."""
import json

import numpy as np
import pandas as pd

from lookahead import analysis, sim


def test_end_to_end_on_simulated_features(tmp_path):
    P = sim.simulate_panel(np.random.default_rng(7), 60, 0.4, "coverage")
    items = pd.DataFrame(dict(coin=[f"C{c}" for c in P["coin"]], week_idx=P["week"], year=P["year"], y=P["y"],
                              n_head=np.maximum(1, np.round(P["cov"])).astype(int),
                              week=pd.Timestamp("2021-01-04") + pd.to_timedelta(P["week"] * 7, unit="D"),
                              catalogue=False, cue=np.random.default_rng(0).random(len(P["y"])) < 0.2))
    feat = tmp_path / "features"; feat.mkdir()
    for v, H in zip(sim.VINTAGES, P["H"]):
        np.save(feat / f"PIT-1B_{v}.npy", np.stack([H, H, H], 1).astype(np.float16))
    res = analysis.run(items, "PIT-1B", feat, tmp_path / "results")
    assert res["h1"]["reject_h1"]
    assert res["remedy"]["leak_gap_after"] < res["remedy"]["leak_gap_before"]
    assert (tmp_path / "results" / "event_study.png").exists()
    json.loads((tmp_path / "results" / "results.json").read_text())
