# Milestone 1 — Working Notes

The live site is [`index.html`](index.html) in this folder, published via GitHub
Pages. **Edit the HTML directly** — that is the deliverable. This file is the
status tracker and the place to keep notes that do not belong on the public page.

Every section below is already written into `index.html`. What remains is
everything marked 🟡 or ⬜.

**Status legend:** ✅ ready · 🟡 needs team input · ⬜ not started

---

## A. Introduction — 🟡

### The problem

Every second, millions of packets move across enterprise networks. A small
fraction of them are hostile: port scans mapping a network for weaknesses,
denial-of-service floods, brute-force attempts against web logins, infiltration
traffic from a compromised host. Network intrusion detection is the problem of
telling those apart from ordinary traffic, quickly enough to act.

Deep packet inspection — reading payload contents — is expensive at line rate and
often impossible when traffic is encrypted. So operational detection increasingly
relies on **flow records**: compact per-connection summaries of timing, volume,
direction, and packet size statistics, with no payload. The question this project
investigates is how much of the malicious/benign distinction survives in that
reduced representation, and whether different attack types leave distinguishable
traces in it.

### Stakeholders

- **Network defenders and SOC analysts**, who tune detection thresholds and absorb
  the cost of false positives — every spurious alert consumes analyst attention
  that a real incident needs.
- **Organizations operating internet-facing infrastructure**, for whom undetected
  intrusion means data loss, service disruption, and regulatory exposure.
- **End users**, whose data sits behind those systems and who have no visibility
  into whether detection is working.
- **The security research community**, which relies on shared benchmark datasets
  to compare methods — and is affected when those benchmarks have defects.

### What is already known, and the gap

Flow-based intrusion detection is a mature field, and CICIDS2017 is one of its
most widely used benchmarks — thousands of published papers report detection
results on it, frequently at accuracies above 99%.

That headline number is the gap. Two independent peer-reviewed audits
([Engelen et al. 2021](https://intrusion-detection.distrinet-research.be/WTMC2021/Resources/wtmc2021_Engelen_Troubleshooting.pdf);
[Lanvin et al. 2023](https://doi.org/10.1007/978-3-031-31108-6_2)) found that the
dataset contains flow-construction errors affecting roughly a quarter of all
records, and that labels were assigned by IP address and time window without
verifying that an attack actually executed. Because attacker hosts are fixed
throughout the capture, a model given IP or timestamp features can reach near
perfect accuracy by memorizing *which machine* was the attacker — learning
nothing transferable about attacks at all.

Much published work on this dataset predates those audits or does not account for
them. The unanswered question is therefore not "can a model classify CICIDS2017
accurately" — that is settled and possibly meaningless — but **which of the
apparent distinctions between traffic classes survive once the known artifacts are
removed.**

### Project blueprint

This semester the team will:

1. Characterize the dataset as it actually is — class balance, feature
   distributions, per-day composition, missing and degenerate values.
2. Establish which flow features separate benign from malicious traffic, and
   attack categories from each other, with host identifiers and timestamps
   excluded from the outset.
3. Test whether those separations hold across capture days, including against
   attack types absent from training.
4. Quantify how much the documented labeling defects move the results, by
   comparing outcomes with known-suspect classes retained versus excluded.
5. Report per-class, with the scope limits stated plainly: findings describe a
   five-day simulated capture from July 2017 and a fixed attack catalog.

### Required figure — ⬜

The milestone requires at least one useful visual with a source citation.

**Recommended:** a class-distribution bar chart (log scale) showing record counts
per label across the seven capture files. It is our own figure, so no attribution
is needed; it makes the imbalance immediately legible; and it is the natural
opening exhibit for RQ1. Generate it from the data and save to `figures/`.

**Alternative if the data is not yet in hand:** the testbed network topology
diagram from Sharafaldin et al. (2018), cited to that paper.

---

## B. Research Questions — 🟡

See `README.md` for the current five. Needs team sign-off. RQ5 is the likely cut
if the team prefers four.

---

## C. Team — 🟡 blocked

| Member | Bio | Photo | Links | Role |
| --- | --- | --- | --- | --- |
| Christopher Taylor | ⬜ | ⬜ | optional | ⬜ |
| Luis Echeverry | ⬜ | ⬜ | optional | ⬜ |
| Tyler Garfield | ⬜ | ⬜ | optional | ⬜ |

**This is the only section with a hard external dependency.** Each member writes
their own two-to-three sentence professional bio and supplies a photo.
Professional links are optional and individually chosen.

Proposed role split, to be confirmed:

- **Coordination & documentation** — milestone tracking, website, README
- **Data & reproducibility** — acquisition, cleaning, sampling, environment
- **Analysis & visualization** — exploratory analysis, comparative work, figures

---

## D. Proposal Overview — ✅

**Problem.** Network intrusion detection depends on distinguishing malicious from
benign traffic, and on separating attack types from one another, using flow
records rather than payload inspection. High-volume environments need detection
that is both fast and reliable.

**Goal.** Characterize the flow-level patterns that separate benign from malicious
traffic and attack categories from each other, while assessing how far those
patterns reflect attacker behavior rather than artifacts of dataset construction.

**Data.** CICIDS2017 (Canadian Institute for Cybersecurity): seven CSVs of labeled
bidirectional flow records covering five days of simulated enterprise traffic in
July 2017, ~85 features per flow, roughly 1.05 GB total. Obtained under CIC's
research-use terms after registration; redistribution is not permitted, so the
repository carries acquisition instructions and a small derived sample rather than
the raw files.

**Questions.** See Section B.

**Constraints.** Data volume relative to available compute; severe class imbalance,
with benign flows dominating and Monday benign-only; a fixed 2017 attack catalog
that cannot speak to novel or contemporary attack vectors; and documented labeling
and flow-construction errors that limit which per-class findings can be trusted.

**Change since Milestone 0.** The approved proposal treated source/destination IP
and port as core descriptive features and did not account for published errata on
this dataset. Both have been revised. Host identifiers and timestamps are now
excluded from modeling, because attacker hosts are fixed in this simulation and
including them lets a model memorize machine identity rather than attack
behavior. Label reliability is treated as an explicit analysis question rather
than an assumption. This narrows the scope but makes the findings defensible —
and it converts a hidden threat to validity into a stated one.

---

## Submission checklist

- 🟡 **Publicly accessible website URL** — the Google Sites version is restricted
  to `colorado.edu` accounts and redirects anonymous visitors to a sign-in. The
  GitHub Pages site in this folder replaces it and is public by default; enable it
  under Settings → Pages, then retire or redirect the Google Site so there is only
  one canonical URL.
- 🟡 Shared repository URL with README and initial structure — push and add the
  URL to `README.md` and the site footer
- 🟡 Website includes Introduction, Research Questions, Team, Proposal Overview —
  all four present; Team section awaits bios and photos
- ✅ At least 3 credible background sources cited — see `references.md`
- ✅ No secrets, credentials, or restricted data in the repository

## Open TODOs in `index.html`

Grep for `TODO` — there are three:

1. `TODO(figure)` — replace the placeholder with the class-distribution chart
2. `TODO(bio)` — three bios, photos, and role labels
3. Footer repository URL
