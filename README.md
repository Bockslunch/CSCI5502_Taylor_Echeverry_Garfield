# Distinguishing Malicious from Benign Network Traffic in CICIDS2017

CSCI 5502 — Data Mining — Semester Group Project
University of Colorado Boulder

**Project website:** _(add URL once Pages is enabled — see below)_

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
| Christopher Taylor | _TBD_ |
| Luis Echeverry | _TBD_ |
| Tyler Garfield | _TBD_ |

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

1. **(Descriptive)** How are flow-level features distributed across benign and
   malicious traffic in CICIDS2017, and how severe is the class imbalance across
   attack categories and across the five capture days?
2. **(Comparative)** Which flow features most sharply separate attack categories
   from one another — and does each separation reflect attacker behavior, or an
   artifact of the capture setup such as fixed host identity or flow-termination
   handling?
3. **(Relationship / pattern)** Do attack categories form distinct, internally
   coherent groupings in flow-feature space, or do some categories overlap enough
   that they are not separable at the flow level?
4. **(Predictive)** How well does a benign-vs-malicious distinction learned on one
   capture day transfer to a different day containing attack types not seen during
   training?
5. **(Explanatory)** How much do the documented labeling and flow-construction
   errors in CICIDS2017 change per-class results, when known-suspect classes are
   excluded versus retained?

_Status: draft pending team sign-off. Five is the assignment ceiling; RQ5 is the
most likely cut, and can be folded into RQ2._

---

## Data

**CICIDS2017** — Canadian Institute for Cybersecurity, University of New Brunswick.

Seven CSV files of labeled bidirectional network flow records covering five days
of simulated enterprise traffic (Monday–Friday, July 2017), with approximately 85
features per flow record.

| File | Size |
| --- | --- |
| `Monday-WorkingHours.pcap_ISCX.csv` | 256 MB |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 167 MB |
| `Wednesday-workingHours.pcap_ISCX.csv` | 272 MB |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 88 MB |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 104 MB |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 72 MB |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 97 MB |

**The dataset is not redistributed in this repository.** CIC requires
registration before download and does not permit redistribution. See
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

---

## Repository Structure

```
.
├── data/           Acquisition instructions and a small derived sample.
│                   Raw CICIDS2017 files are gitignored — never commit them.
├── notebooks/      Exploratory analysis notebooks.
├── src/            Reusable code: loading, cleaning, feature handling.
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
| 1 | Project framing & foundation — research questions, background, team, repository, website | In progress |
| 2 | _Fill in from Canvas_ | Not started |
| 3 | _Fill in from Canvas_ | Not started |
| 4 | _Fill in from Canvas_ | Not started |

> Replace milestones 2–4 with the actual titles and due dates from Canvas.

---

## Reproducibility

Nothing in this repository requires the raw dataset to be present in order to be
read and understood. To run the analysis code:

1. Follow [`data/README.md`](data/README.md) to obtain CICIDS2017.
2. Place the CSV files in `data/raw/` (gitignored).
3. Create the environment and run notebooks from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## References

See [`docs/references.md`](docs/references.md) for the full reference list.
