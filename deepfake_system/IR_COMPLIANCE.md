# IR compliance — requirement traceability

Every requirement in the Investigation Report mapped to where it is
implemented, with the gaps that were closed in this revision marked
**FIXED**. This table is also usable as an appendix in your final report.

---

## Chapter 1.4 — Objectives

| Objective | Status | Where |
|---|---|---|
| Literature review of generation and detection | Report only | Chapter 2 |
| Train a CNN on frame-level irregularities (warping, flaws, digital noise) | Met | `train.py`, `models/net.py` |
| **MesoNet architecture (Afchar et al., 2018)** as cited | **FIXED** | `models/net.py` → `MesoInception4`, select with `--backbone mesonet` |
| Functional Python prototype: upload, run, report | Met | `app/server.py`, `app/static/` |
| Evaluate on FaceForensics++ and Celeb-DF | Met | `data/build_manifest.py`, `evaluate.py` |
| Report Accuracy, Precision, Recall, F1 | Met | `evaluate.py` → `metrics()` |

The report cites MesoNet, but a 12k-parameter network from 2018 will not
reach the accuracy you want. Both are now available on the same data and
protocol, which turns the discrepancy into an ablation study:

```bash
python train.py --backbone mesonet          --epochs 22   # the cited network
python train.py --backbone tf_efficientnetv2_s --epochs 22 # the modern one
```

Report both rows. "I implemented the cited architecture, measured it, and
justified moving to a stronger backbone" is a better answer than either
silently substituting or silently accepting a weak result.

---

## Chapter 1.5 — Scope

| Scope item | Status | Where |
|---|---|---|
| Frame-by-frame analysis | Met | `app/engine.py` → `FaceExtractor.extract` |
| OpenCV for video processing | Met | throughout |
| MTCNN face detection | Met | `FaceExtractor`, with MediaPipe and Haar fallbacks |
| CNN detecting high-frequency noise and up-sampling artifacts | Met | SRM stream in `models/net.py`; `spectral_slope` in `app/heuristic.py` |
| Aggregated confidence score ("95% Fake") | Met | trimmed-mean pooling in `engine.analyse` |
| Uploaded files only, no live stream | Met | matches the stated limitation |
| Target users: forensic analysts, moderators, public | Met | interface is built for this |

---

## Chapter 3.4.1 — Functional requirements

| Requirement | Status | Where |
|---|---|---|
| **Multimodal: visual AND audio (voice cloning, lip-sync)** | **FIXED** | `app/audio.py` — was entirely absent |
| **Metadata and provenance: watermarks, authenticity signatures** | **FIXED** | `app/provenance.py` — C2PA/JUMBF, XMP, generator fingerprints |
| Platform independence (third-party, not platform-native) | Met | self-hosted, no platform APIs |
| Automated flagging with a final classification | Met | `label` in every response |

### The audio branch

Two measurements, reported separately because they answer different
questions:

- **Lip-sync agreement** — correlation between mouth openness (from the
  face crops the visual pipeline already produced) and the audio
  envelope, searched over a ±0.4 s lag window. Verified on synthetic
  test clips: **r = 0.94** with matched audio, **r = 0.34** with
  unrelated audio on the same video.
- **Synthetic voice indicators** — high-band spectral flatness, noise
  floor, and envelope regularity.

A caveat worth stating in the viva before an examiner raises it: Wav2Lip
and SadTalker *drive the mouth from the audio*, so they score **high** on
lip-sync agreement. This is a mismatch detector, not a fake detector. It
catches dubbing, re-voicing, and swaps where the audio came from
elsewhere. That is why a mismatch only widens the confidence margin
rather than pushing the score toward "manipulated".

### Provenance

C2PA is the real answer to "embedded authenticity signatures". The module
detects the JUMBF box, and validates the manifest properly when the
`c2pa` package is installed. States: `verified`, `invalid`,
`present_unverified`, `absent`. Provenance is kept strictly separate from
the manipulation score — where a file came from is a different question
from whether the face was edited.

---

## Chapter 3.4.1 — Non-functional requirements

| Requirement | Status | Where |
|---|---|---|
| High forensic accuracy, low false positives (82.1% of users) | Partly | three-state output; the real number needs training |
| **Speed / near real-time (66.1%)** | **FIXED** | per-stage `timings` in every response |
| **Scalability and load capacity** | **FIXED** | see below — this was a real bug |
| Ongoing algorithmic adaptation | Met | swap `best.pt`, restart; `model_version` recorded per report |

### The scalability bug

`analyse` was declared `async def` but called blocking CPU work directly.
In asyncio that occupies the event loop, so **one upload froze every
other request**, including the health check. Now it runs through
`run_in_threadpool` behind a semaphore (`DF_CONCURRENCY`, default 2).

Measured with four simultaneous uploads:

| | `/api/status` latency under load |
|---|---|
| Before | blocked for the full analysis (~12 s) |
| After | 1–15 ms |

Worth putting in your report as a named defect found and fixed — that is
exactly the kind of thing a testing chapter is for.

---

## Chapter 3.4.1 — UI/UX requirements

| Requirement | Status | Where |
|---|---|---|
| **Browser extension or mobile format (76.8%)** | **FIXED** | `extension/` — Manifest V3 |
| **Works on Instagram, Facebook, TikTok, X** | **FIXED** | context menu on any `<video>` element |
| Simple explainable output with visual cues (XAI) | Met | frame strip, margin gauge, signals panel |
| "A definite Real or Fake tag" | **Deliberate deviation** | see below |

### The extension

Right-click any video → *Check this video for manipulation*. It fetches
the media in the page's own context (so session cookies apply), posts it
to **your** server, and shows the verdict in a notification. No third
party ever sees the video.

Load it with `chrome://extensions` → Developer mode → *Load unpacked* →
select `extension/`, then set your server address in its options page.

Known limit, and state it rather than hide it: sites that stream via MSE
(`blob:` URLs — TikTok and Instagram often do) cannot have the media
pulled from the page. The extension detects this and says so instead of
failing silently.

### The one deliberate deviation

Your report specifies "a definite Real or Fake tag". The system defaults
to three states, adding `inconclusive` when the confidence margin covers
the decision threshold.

This is a considered deviation, not an oversight. Requirement 1 asks for
a high detection rate **and** low false positives, and those pull against
each other at exactly the point where the margin straddles the line.
Forcing a call there manufactures a false positive or a false negative
roughly half the time. A forensic tool that says "I do not know" is more
useful — and more defensible — than one that guesses.

If your supervisor wants literal compliance, it is one flag:

```python
# config.py
INFER.strict_binary = True
```

Verified: with `strict_binary=False` a straddling case returns
`inconclusive`; with `True` it returns a binary label plus a note that
confidence is low. Either way `margin_straddles_threshold` is in the
response, so nothing is hidden.

---

## Chapter 3.4.1 — Security and privacy

| Requirement | Status | Where |
|---|---|---|
| Zero-retention: files not stored, retained, or used for training | Met | temp file deleted in a `finally` block; `PrivateTmp=true` |
| **Encrypted transport between UI and backend** | **FIXED** | `deploy/nginx.conf` — TLS 1.2/1.3, HSTS, CSP |

Findings **are** stored — the verdict, the SHA-256 of the evidence, and
the model version. Never the video, never the thumbnails. A forensic tool
that cannot reproduce the report it issued last week is not much use;
zero-retention covers the media, not the conclusion.

Storing the hash gives you duplicate detection for free: re-upload the
same file and the report tells you it has been seen before, and when.
Verified working.

---

## Chapter 2.4 — Technical stack

| Item | Status | Where |
|---|---|---|
| 2.4.2 Python | Met | throughout |
| 2.4.3 VS Code | N/A | your editor, not the artefact |
| **2.4.4 AWS DynamoDB** | **FIXED** | `app/store.py` — was missing entirely |
| 2.4.5 Tools and libraries | Met | `requirements.txt` |
| 2.4.6 Ubuntu 22.04 LTS | Met | `deploy/deepfake.service` |
| **2.4.7 Nginx** | **FIXED** | `deploy/nginx.conf` |

`app/store.py` uses DynamoDB when boto3 and credentials are present, and
falls back to a local JSON-lines file otherwise — so it runs on your
laptop and on AWS with no code change. `/api/status` reports which
backend is live, so you can demonstrate both.

Do not commit AWS keys. Use an instance role scoped to `PutItem`,
`GetItem` and `Scan` on the one table. Table creation command is in
`deploy/README.md`.

---

## Still open

Two things the report asks for that are **not** done, listed so you are
not surprised by them:

1. **A trained checkpoint.** The system runs on the classical baseline
   until you train. This is the single biggest remaining item, and the
   accuracy claims in Chapter 4 depend on it entirely.
2. **A mobile app.** The UI/UX requirement says "browser extension **or**
   a feature of a mobile application". The extension satisfies the
   disjunction. The web interface is responsive, so a mobile browser
   works, but there is no native app and you should not claim one.

---

## New API surface

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | engine, store backend, model version, branches enabled |
| `POST /api/analyse` | now also returns `audio`, `content_credentials`, `timings`, `model_version`, `report_id`, `seen_before` |
| `GET /api/reports?limit=25` | past findings |
| `GET /api/reports/by-hash/{sha256}` | every report for one file |

All additions are new keys. Nothing existing was renamed or removed, so
your rebuilt front end keeps working untouched.
