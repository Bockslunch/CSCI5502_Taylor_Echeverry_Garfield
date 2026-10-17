# Distinguishing Malicious from Benign Network Traffic in CICIDS2017

CSCI 5502 — Data Mining — Semester Group Project
University of Colorado Boulder

**Project website:** <https://bockslunch.github.io/CSCI5502_Taylor_Echeverry_Garfield/>

The site lives in [`docs/`](docs/) and is published with GitHub Pages. To turn it
on: **Settings → Pages → Source: "Deploy from a branch" → Branch: `main`,
Folder: `/docs` → Save.** The URL appears on that page within a minute or two and
is publicly accessible with no sign-in.

To preview locally before pushing:

```bash
python -m http.server -d docs 8000   # then open http://localhost:8000
```

---

## Team

| Member | Role (initial) |
| --- | --- |
| Christopher Taylor | Coordination · Data · Analysis |
| Luis Echeverry | Data · Reproducibility |
| Tyler Garfield | Analysis · Visualization |

Suggested role split — adjust before submitting. Roles may overlap and may change.

- **Coordination & documentation** — milestone tracking, website content, README upkeep
- **Data & reproducibility** — acquisition, cleaning, sampling, environment, repo hygiene
- **Analysis & visualization** — exploratory analysis, comparative work, figures

> Contact information is intentionally omitted from this public repository. Add
> GitHub or LinkedIn links here only if each member chooses to share them.

---

## Project Goal

Characterize the flow-level patterns that separate benign from malicious network
traffic, and that separate attack categories from one another, in a labeled
benchmark dataset — while assessing how far those patterns reflect attacker
behavior rather than artifacts of how the dataset was constructed.

The project is framed around a detection problem, not around a particular
modeling technique. Method selection is deferred to later milestones.

---

## Research Questions

Revised after the Milestone 2 exploratory analysis; the reasons for each change are
in the Milestone 2 report (Section 6).

1. **(Descriptive)** How are flow-level features distributed across benign traffic
   and each attack class, how severe is the class imbalance, and how do the known
   data-quality issues (duplicates, the 120-second flow timeout, the −1 window
   code, zero-payload flows) vary by class?
2. **(Comparative)** Excluding destination port, which flow features separate the
   attack categories that have enough records to compare (the 11 attack classes
   with at least 600 flows), and which separations survive when artifact-prone
   features (`Init_Win_bytes_*`, durations at the 120-second cap) are removed?
3. **(Relationship / pattern)** Do the four behavioral groups seen in the
   exploratory analysis hold up when individual flows are clustered, and are the
   overlapping pairs (Web Attack – Brute Force vs. XSS; Bot vs. BENIGN) separable
   at the flow level at all?
4. **(Predictive)** Using leave-one-day-out evaluation on deduplicated data, how
   well does a benign-vs-malicious model detect the held-out day's attacks, all of
   which are unseen types, and how much of any drop comes from shifted benign
   traffic rather than from the new attacks?
5. **(Explanatory)** How much do per-class results change when flows with
   identifiable defects (zero-payload web-attack flows, flows at the 120-second
   cap, feature vectors shared across labels) are excluded versus retained?

_Status: revised at Milestone 2, pending team sign-off._

---

## Data

**CICIDS2017** — Canadian Institute for Cybersecurity, University of New Brunswick.

Eight CSV files (`MachineLearningCSV.zip`) of labeled bidirectional network flow
records covering five days of simulated enterprise traffic (Monday–Friday, July
3–7, 2017): 2,830,743 flows × 79 columns (78 flow features plus `Label`).

| File | Size | Flows |
| --- | --- | --- |
| `Monday-WorkingHours.pcap_ISCX.csv` | 177 MB | 529,918 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 135 MB | 445,909 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 225 MB | 692,703 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 52 MB | 170,366 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 83 MB | 288,602 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 58 MB | 191,033 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 77 MB | 286,467 |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 77 MB | 225,745 |

After cleaning (`notebooks/01-data-cleaning.ipynb`): 2,520,656 flows × 72 columns.

**The dataset is not redistributed in this repository.** CIC grants access through
a registration form and asks users to cite Sharafaldin et al. (2018); it states no
license and grants no redistribution right, so we do not redistribute the data. See
[`data/README.md`](data/README.md) for acquisition instructions.

### Known data quality issues

This dataset has documented defects that shape our methodology. Two peer-reviewed
analyses (Engelen et al. 2021; Lanvin et al. 2023 — see
[`docs/references.md`](docs/references.md)) identify flow-construction errors
affecting roughly a quarter of all records, and labeling performed by IP address
and time window without verifying that an attack actually executed.

Consequences we carry through the analysis:

- Source/destination IP and timestamp features are **excluded from modeling**.
  Attacker hosts are fixed in this simulation, so models trained on these features
  memorize machine identity rather than attack behavior.
- Results are reported **per class**, not as aggregate accuracy.
- Findings are scoped to the attack catalog captured in July 2017 and are not
  claimed to generalize to contemporary traffic.
- Destination port is kept for description but **excluded from modeling**: 10 of
  the 14 attack classes use a single port, so it encodes the testbed's layout.
- `Init_Win_bytes_*` = −1 marks flows with no TCP window (non-TCP traffic) and is
  treated as a category, not a size. Flow Duration is capped by CICFlowMeter's
  120-second timeout and treated as right-censored.

---

## Repository Structure

```
.
├── data/           Acquisition instructions and a small derived sample.
│                   Raw CICIDS2017 files are gitignored — never commit them.
├── notebooks/      01-data-cleaning.ipynb (raw → cleaned), 02-eda.ipynb (exploration).
├── src/            Scripts: cleaning replay, quality statistics, EDA figures.
├── figures/        Generated figures referenced by the website and report.
├── docs/           The project website (GitHub Pages) plus project documentation.
│   ├── index.html      The site itself — all four required sections
│   ├── assets/         Site stylesheet
│   ├── milestone-1.md  Working notes and section status tracker
│   └── references.md   Canonical reference list
└── README.md
```

---

## Milestone Roadmap

| Milestone | Focus | Status |
| --- | --- | --- |
| 0 | Project idea and proposal | Approved |
| 1 | Project framing & foundation — research questions, background, team, repository, website | Complete |
| 2 | Data acquisition, preparation & exploration — provenance, cleaning, data dictionary, quality assessment, EDA | In progress |
| 3 | _Fill in from Canvas_ | Not started |
| 4 | _Fill in from Canvas_ | Not started |

> Replace milestones 3–4 with the actual titles and due dates from Canvas.

---

## Reproducibility

Nothing in this repository requires the raw dataset to be present in order to be
read and understood. To run the analysis code:

1. Follow [`data/README.md`](data/README.md) to obtain CICIDS2017 and place the
   eight CSV files in `data/raw/` (gitignored).
2. Create the environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the notebooks top to bottom from the `notebooks/` folder (their paths are
   relative to it):
   - `01-data-cleaning.ipynb` → writes `data/processed/cicids2017_clean.csv`
   - `02-eda.ipynb` → exploratory analysis of the cleaned data
4. Regenerate the Milestone 2 statistics and figures from the repository root:

   ```bash
   python src/m2_rebuild_clean.py data/raw data/processed   # replays notebook 01, logs every step
   python src/m2_quality_stats.py data/processed            # Section 4–5 numbers → m2_stats.json
   python src/m2_eda_figures.py data/processed figures docs/assets   # Figures 1–7, light + dark, copied to the site
   ```

---

## References

See [`docs/references.md`](docs/references.md) for the full reference list.
