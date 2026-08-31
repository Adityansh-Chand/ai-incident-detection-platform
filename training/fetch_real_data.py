"""Fetch the Server Machine Dataset -- real telemetry, real incidents.

`training/generate_timeseries.py` produces incidents we designed: we chose the
escalation shape, the correlation between latency and CPU, the recovery curve.
A detector doing well on it has recovered our own assumptions about what an
outage looks like.

SMD is five weeks of real operational telemetry from 28 server machines in three
data centres, 38 metrics each, sampled every minute. Its anomalies were labelled
by the operators **from actual incident reports** -- which is exactly the ground
truth the synthetic track cannot have.

    Su, Zhao, Niu, Liu, Sun and Pei, "Robust Anomaly Detection for Multivariate
    Time Series through Stochastic Recurrent Neural Networks", KDD 2019.
    https://github.com/NetManAIOps/OmniAnomaly
    Released under the MIT licence.

**Machine selection is fixed and stated up front: the first machine of each of
the three groups.** All 28 would be ~550MB. Choosing which machines to report
*after* seeing the results would be cherry-picking, so the rule is declared here
and never revisited -- if a chosen machine scores badly, that score is published.

Cached under datasets/real/ and NOT committed.

    python training/fetch_real_data.py           # download and cache
    python training/fetch_real_data.py --check   # verify the cache, no network
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "datasets" / "real"

BASE = ("https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly"
        "/master/ServerMachineDataset")

# First machine of each group. See the docstring: this rule is fixed in advance.
MACHINES = ["machine-1-1", "machine-2-1", "machine-3-1"]
PARTS = ["train", "test", "test_label"]

# Filled in on first download and asserted afterwards, so a changed upstream file
# surfaces here rather than as quietly different metrics.
CHECKSUMS = {
    "train/machine-1-1": "ecebd2933ccdc1126e2abbfb538762dbde16410cebc77257a7b96e1e1fbcf753",
    "test/machine-1-1": "bdb801d8d52ce6b0c4a20c310edf31b5cee9bc82eb84fb38e7d4a19309b16ddb",
    "test_label/machine-1-1": "cddf33441ae480c6074e173c897d4715eb893f24a7078d3d3e1b79b200c7cd19",
    "train/machine-2-1": "d6eb13ff74a537cf33686319fffdd4cdcb394862570a3acd519b4c37af97cd2e",
    "test/machine-2-1": "2d103661799271be958227907950a1c76803356442d60a4070671cc0abbefb20",
    "test_label/machine-2-1": "416dd20a447f38b8d9394382293ed44f509c1c3f99c4aa2fbff179498626dbdc",
    "train/machine-3-1": "c1186ac085809c0590467dd42f4f12180012ebd994c1e615250dbed2a3588111",
    "test/machine-3-1": "b110b204ba4919250351eba7aa253d73ab54af0cf18e272b9b284de5fa7df88d",
    "test_label/machine-3-1": "549348a9c63ed922e85157885bcb21289a8c9f5af94b50147a21cb7b928220ec",
}


def paths_for(machine):
    return {part: CACHE_DIR / part / f"{machine}.txt" for part in PARTS}


def download():
    for part in PARTS:
        (CACHE_DIR / part).mkdir(parents=True, exist_ok=True)

    digests = {}
    for machine in MACHINES:
        for part, path in paths_for(machine).items():
            url = f"{BASE}/{part}/{machine}.txt"
            print(f"downloading {part}/{machine}.txt")
            request = urllib.request.Request(
                url, headers={"User-Agent": "portfolio-fetch"}
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                data = response.read()
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            digests[f"{part}/{machine}"] = digest
            rows = data.count(b"\n")
            print(f"  {len(data) / 1e6:.1f} MB, {rows} rows, sha256 {digest[:16]}...")

    if CHECKSUMS:
        mismatched = [k for k, v in digests.items() if CHECKSUMS.get(k) != v]
        if mismatched:
            print(f"FAIL: checksum mismatch for {mismatched}")
            return 1
    else:
        print("\nno checksums recorded yet; paste these into CHECKSUMS:")
        for key, digest in digests.items():
            print(f'    "{key}": "{digest}",')
    return 0


def check():
    problems = []
    for machine in MACHINES:
        for part, path in paths_for(machine).items():
            key = f"{part}/{machine}"
            if not path.exists():
                problems.append(f"MISSING {path}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = CHECKSUMS.get(key)
            status = "ok" if expected == digest else "UNVERIFIED" if not expected else "MISMATCH"
            print(f"{key:28s} {digest[:16]}... {status}")
            if expected and expected != digest:
                problems.append(f"{key}: checksum mismatch")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        print("run: python training/fetch_real_data.py")
        return 1
    if not CHECKSUMS:
        print("FAIL: no checksums recorded")
        return 1
    print(f"OK: SMD present for {len(MACHINES)} machines, checksums match")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the cached copy without downloading")
    args = parser.parse_args()
    return check() if args.check else download()


if __name__ == "__main__":
    sys.exit(main())
