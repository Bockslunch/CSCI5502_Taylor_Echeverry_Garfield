# References

Citation style: **IEEE**. Keep this file as the single source of truth for the
project's reference list; the website's background section should mirror it.

---

## Background / Related Work

### [1] The dataset itself

I. Sharafaldin, A. Habibi Lashkari, and A. A. Ghorbani, "Toward Generating a New
Intrusion Detection Dataset and Intrusion Traffic Characterization," in *Proc.
4th Int. Conf. Information Systems Security and Privacy (ICISSP)*, Funchal,
Portugal, Jan. 2018, pp. 108–116.
<https://www.scitepress.org/papers/2018/66398/66398.pdf>

**What it contributes.** The primary source for CICIDS2017 — how the testbed was
built, which attacks were executed on which days, how benign background traffic
was generated from abstracted user profiles, and how the ~85 flow features were
extracted with CICFlowMeter. It also introduces the authors' own framework for
evaluating IDS datasets against eleven criteria. We rely on this paper for the
ground truth of what each label is supposed to mean and for the intended
composition of each capture day. Read critically: it is the authors' account of
their own dataset, and the two sources below dispute parts of it.

### [2] Documented defects in the dataset

G. Engelen, V. Rimmer, and W. Joosen, "Troubleshooting an Intrusion Detection
Dataset: the CICIDS2017 Case Study," in *Proc. IEEE Security and Privacy
Workshops (SPW) — Workshop on Traffic Measurements for Cybersecurity (WTMC)*,
2021, pp. 7–12.
<https://intrusion-detection.distrinet-research.be/WTMC2021/Resources/wtmc2021_Engelen_Troubleshooting.pdf>

**What it contributes.** A systematic audit of CICIDS2017 that reconstructs the
dataset from the original PCAPs. Three findings drive our methodology. First, the
flow-construction bug: CICFlowMeter closed flows on a single FIN rather than a
bidirectional teardown, generating spurious trailing flows that inherited attack
labels — 25.9% of the dataset, up to 50% in some attack classes. Second, the
labeling strategy used only source/destination IP and time windows, never
verifying that an attack executed; several Web Attack classes turn out to be
connections with no forward data transfer. Third, the authors warn explicitly
about shortcut learning from host identifiers and timestamps, since attacker
hosts are fixed. Their recommendations — inspect feature importance as a sanity
check, report per-class rather than aggregate performance — are adopted directly
in this project.

### [3] Independent confirmation, and the size of the effect

M. Lanvin, P.-F. Gimenez, Y. Han, F. Majorczyk, L. Mé, and É. Totel, "Errors in
the CICIDS2017 Dataset and the Significant Differences in Detection Performances
It Makes," in *Risks and Security of Internet and Systems (CRiSIS 2022)*, Lecture
Notes in Computer Science, vol. 13857. Cham, Switzerland: Springer, 2023,
pp. 18–33. <https://doi.org/10.1007/978-3-031-31108-6_2>

**What it contributes.** An independent team reaching compatible conclusions by a
different route, which matters — it means [2] is not an idiosyncratic reading.
This paper documents packet misordering, packet duplication, and attacks that
were executed but never correctly labeled, then quantifies how much supervised
detection performance shifts once the corrections are applied. It supplies the
justification for RQ5: the difference is large enough that results computed on
the uncorrected CSVs cannot be reported without qualification.

---

## Tools and code referenced

G. Engelen, "CICFlowMeter" (patched fork), GitHub repository.
<https://github.com/GintsEngelen/CICFlowMeter>

G. Engelen, "WTMC2021-Code," GitHub repository.
<https://github.com/GintsEngelen/WTMC2021-Code>

> Note: neither repository publishes corrected CSV files. Reproducing corrected
> data requires the raw PCAPs and a full re-run of feature extraction.

---

## Adding sources

The milestone requires at least three credible sources, at least two of which are
scholarly, technical, governmental, or otherwise authoritative. All three above
meet the authoritative bar, so the requirement is satisfied — additional sources
should be added only when they do real work for the project.

When adding one, include a short paragraph on **what it contributes to our
understanding**, not just the citation. Listing citations without synthesis is
called out in the grading criteria.
