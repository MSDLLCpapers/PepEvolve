import pandas as pd
import numpy as np
import pickle as pkl
import os
import argparse
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

base_dir = os.path.join(os.path.dirname(__file__), "../../../output/journal/result")

mfpgen = rdFingerprintGenerator.GetMorganGenerator()


def diversity_score(fps):
    fps = list(fps)
    n = len(fps)
    if n < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += float(np.sum(sims))
        count += len(sims)
    return 1.0 - total / count


def compute_diversity_for_step(step, fps_list):
    return {"Step": step, "diversity": diversity_score(fps_list)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    label, mode = args.label, args.mode

    run_dfs = []
    for run in [1, 2, 3]:
        folder = os.path.join(base_dir, f"{mode}{run}")
        if mode.startswith("pepinvent"):
            csv_path = os.path.join(folder, "results.csv")
        elif mode.startswith("traditional_ga"):
            csv_path = os.path.join(folder, "results_evolving_step1000.csv")
        else:
            csv_path = os.path.join(folder, "results_evolving_step250.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["run"] = run
            run_dfs.append(df)
            print(f"  Loaded: {csv_path} ({len(df)} rows)")
        else:
            print(f"  Missing: {csv_path}")

    if not run_dfs:
        print(f"No data for {label}")
        return

    combined = pd.concat(run_dfs, ignore_index=True)
    print(f"Computing fingerprints for {len(combined)} molecules...")
    combined["mol"] = combined["SMILES"].apply(Chem.MolFromSmiles)
    combined.dropna(subset="mol", inplace=True)
    combined["morgan_fps"] = combined["mol"].apply(mfpgen.GetFingerprint)

    # Compute diversity per run per step (enables mean +/- std across replicates)
    step_run_groups = [(step, run, df_grp["morgan_fps"].tolist())
                       for (step, run), df_grp in combined.groupby(["Step", "run"])]
    print(f"Computing diversity for {len(step_run_groups)} (step, run) groups (n_jobs={args.n_jobs})...")

    def _div_for_step_run(step, run, fps_list):
        return {"Step": step, "run": run, "diversity": diversity_score(fps_list)}

    records = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_div_for_step_run)(step, run, fps_list)
        for step, run, fps_list in step_run_groups
    )

    df_div = pd.DataFrame(records).sort_values(["Step", "run"]).reset_index(drop=True)

    out_div = f"output/{label}_diversity.pkl"
    os.makedirs("output", exist_ok=True)

    with open(out_div, "wb") as f:
        pkl.dump(df_div, f)

    print(f"Saved {out_div} ({len(df_div)} steps)")


if __name__ == "__main__":
    main()
