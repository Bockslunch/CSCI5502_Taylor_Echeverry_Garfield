# Data

**The CICIDS2017 dataset is not included in this repository.**

The Canadian Institute for Cybersecurity (CIC) requires registration before
download and does not permit redistribution. Committing the raw files here would
violate those terms and would also push roughly 1 GB of data into version
control. The raw files are excluded by `.gitignore`.

---

## Acquisition

1. Go to the CIC IDS 2017 dataset page:
   <https://www.unb.ca/cic/datasets/ids-2017.html>
2. Complete the download form with your name, email, and institution. Access is
   granted to researchers, including university-affiliated students.
3. Download **`MachineLearningCSV.zip`** — the labeled flow records used by this
   project. (`GeneratedLabelledFlows.zip` and the raw PCAPs are not required
   unless we later reproduce the corrected feature extraction described below.)
4. Extract the seven CSV files into `data/raw/`.

Expected result:

```
data/raw/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

## Citation requirement

CIC asks that the dataset be cited in any work that uses it:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani. "Toward
> Generating a New Intrusion Detection Dataset and Intrusion Traffic
> Characterization." *4th International Conference on Information Systems
> Security and Privacy (ICISSP)*, Portugal, January 2018.

---

## Directory layout

| Path | Contents | Tracked? |
| --- | --- | --- |
| `data/raw/` | Original CICIDS2017 CSVs, exactly as downloaded | No |
| `data/interim/` | Intermediate outputs from cleaning steps | No |
| `data/processed/` | Analysis-ready datasets | No |
| `data/sample/` | Small non-sensitive sample for inspection and CI | **Yes** |

Treat `data/raw/` as read-only. Every transformation should be reproducible from
raw by running code in this repository — never by hand-editing a file.

---

## Committed sample

`data/sample/` holds a small excerpt so that anyone cloning this repository can
inspect the schema and run code without first obtaining the full dataset.

Keep it small (a few thousand rows), keep it class-balanced enough to be useful,
and generate it with a script so it is reproducible. The sample contains no
credentials and no real personal data — CICIDS2017 traffic is synthetically
generated in a testbed.

---

## Known issues with this dataset

Two peer-reviewed analyses document substantive defects. These are not optional
background reading; they shape how the data must be handled.

- **Flow construction.** CICFlowMeter terminated flows on a single FIN packet
  rather than a bidirectional close, producing spurious trailing flows that were
  inherited attack labels. Engelen et al. report these account for ~25.9% of the
  dataset, and up to 50% within some attack classes.
- **Labeling.** Labels were assigned by source/destination IP plus time window,
  without verifying that an attack actually executed. Several Web Attack classes
  consist largely of flows with no forward data transfer — connections where the
  payload never ran.
- **Attack simulation.** The DoS Hulk tool used was deprecated and sent incorrect
  HTTP headers, so the intended denial-of-service effect was not produced.
- **Shortcut features.** Attacker hosts are fixed throughout the capture, so IP
  address and timestamp features let a model identify attacks by machine identity
  rather than behavior. **Exclude them from modeling.**

Corrected extraction is possible via the authors' patched CICFlowMeter
(<https://github.com/GintsEngelen/CICFlowMeter>) and labeling scripts
(<https://github.com/GintsEngelen/WTMC2021-Code>), but **no corrected CSVs are
published** — regenerating them requires downloading the raw PCAPs and
re-running the full extraction pipeline. This is out of scope for the semester
unless we decide otherwise; see `docs/references.md`.
