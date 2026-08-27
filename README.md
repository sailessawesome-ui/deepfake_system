[Uploading README.md…]()
# Frame Zero — deepfake video verification

A working web app plus the training pipeline behind it. Built against the
IR spec: MTCNN face extraction, CNN classifier, frame-by-frame analysis,
FF++ / Celeb-DF evaluation, explainable output, zero-retention privacy.

**It runs before you have trained anything.** With no checkpoint present
it falls back to a signal-based classical detector, clearly labelled as
such in the interface. Train a model, drop `best.pt` in place, restart —
the interface is identical and the header chip flips from
`baseline · no checkpoint` to `model · tf_efficientnetv2_s`.

---

## Requirement compliance

`IR_COMPLIANCE.md` maps every requirement in the Investigation Report to
where it is implemented, and flags the one deliberate deviation. Read it
before the viva — it doubles as a report appendix.

## Run it now

```bash
cd deepfake_system
./run.sh                    # http://127.0.0.1:8000
./run.sh 0.0.0.0 8080       # reachable from your network
```

That installs five packages — FastAPI, uvicorn, numpy, OpenCV,
python-multipart. No torch needed until you train.

Manually, if you prefer:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-web.txt
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Install `ffmpeg` too if you can (`sudo apt install ffmpeg`). Without it
the app still works, but codec, bitrate and metadata detection go blank,
and those drive the messenger-provenance flags.

### Testing it

Send a few videos through WhatsApp to yourself, download them back, and
drop those in. They arrive named `VID-20260825-WA0007.mp4` with metadata
stripped and a capped resolution, so the provenance flags light up and
the confidence margin widens. That is the point of the interface, and it
demos well.

---

## What the interface shows

**The margin gauge** is the centrepiece. It plots the score, the margin
around it, and the decision line on one 0–1 axis. When the margin covers
the line the verdict is `Inconclusive` and the tool says so instead of
guessing. Your requirement was a high detection rate *and* low false
positives — a three-state output is how you get both, and it is a better
answer in a viva than a forced binary.

**The frame strip** is the explainability requirement made literal: the
actual tracked face crops, each with a score bar tinted from teal to
rose, above a score-over-time plot with the threshold drawn in. If a
manipulation affects only part of a clip it shows up as a peak.

**Signals** breaks down what drove the score (classical engine only).
**File and origin** shows codec, bitrate and the five provenance flags.

Colour is tri-state and semantic throughout — teal, amber, rose map one
to one onto authentic, inconclusive, manipulated. Hue is never
decoration.

---

## What to expect from the numbers

Read this before you write a target into your report.

**99% accuracy and 99% F1 are achievable on an in-distribution test
split.** Train on FF++ c23 and Celeb-DF-v2, test on held-out videos from
those same datasets, aggregate to video level. 97–99% is a normal result.

**The same model on a WhatsApp forward will not be at 99%, and no
published system is.**

| What changes | Typical effect |
|---|---|
| Same datasets, held-out videos | baseline, 97–99% video-level |
| A dataset you did not train on (FF++ → Celeb-DF) | roughly 65–75% AUC for most published methods |
| A generator family you never trained on | similar or worse |
| Heavy re-compression (c40, or a messenger transcode) | another 5–15 points off |
| All three at once — "a WhatsApp video from a stranger" | the open research problem |

WhatsApp re-encodes every video it delivers: H.264 around 1 Mbps,
resolution capped at 480p on many clients, 4:2:0 chroma, metadata
stripped. The high-frequency traces a CNN keys on are the first thing a
low-bitrate encoder discards. A forwarded video has been through that
twice.

The framing that holds up under questioning:

> The system reaches ≥99% accuracy and F1 at video level on the held-out
> in-distribution test set, retains X% under messenger transcoding, and
> Y% on unseen generator families. The gap is characterised rather than
> hidden.

That reads as competent. An unqualified "99% on any video" takes one
examiner with a phone to disprove in the room.

The techniques that actually narrow the gap, all implemented here:

1. **Training on messenger-degraded views** (`data/degrade.py`) — the
   biggest single lever.
2. **A clean/degraded consistency loss** (`train.py`) — the same clip is
   shown both ways and the model is penalised for disagreeing with
   itself, so it cannot use codec artifacts as a shortcut. This is your
   novelty claim.
3. **A high-frequency SRM stream** beside the RGB backbone
   (`models/net.py`), which degrades more gracefully than RGB alone.
4. **Video-level aggregation with a trimmed mean** — twelve clips voting
   beats one, usually by several points.
5. **Threshold and temperature calibration** on validation, never test.
6. **The `inconclusive` verdict** described above.

---

## Your four datasets

| Archive | Role |
|---|---|
| `FF++` (6.9 GB) | main training set — the four classic manipulations |
| `celebdf_faces` (1.9 GB) | training + in-distribution test, higher visual quality |
| `df40_frames` (1.4 GB) | this is what covers **AI fusion** — 40 generator families including diffusion and face-fusion methods |
| `wild` (1.1 GB) | **hold out entirely.** Never train on it. Your honest generalisation number. |

DF40 is the important one for AI-fusion coverage. Face-swap detectors
trained only on FF++ do badly on diffusion- and fusion-generated faces
because the artifact families differ. Training across DF40's variety buys
coverage; holding a few of its families out
(`config.DATA.holdout_methods`) lets you prove generalisation instead of
asserting it.

---

## Training

```bash
pip install -r requirements.txt

python -m data.build_manifest --dry-run   # check the detected layout first
python -m data.build_manifest             # write the manifest

python train.py --epochs 22 --batch-size 8

python evaluate.py --checkpoint runs/v1/best.pt --split test
python evaluate.py --checkpoint runs/v1/best.pt --split holdout

./scripts/make_messenger_testset.sh ./test_videos ./messenger_eval
```

Then copy `best.pt` and `calibration.json` into `runs/v1/` and restart
the server. Nothing else changes.

Splits are assigned **by identity**, not by frame. If you split frames
randomly you will see 99.9% and it will mean nothing, because the same
face is in train and test. That is the most common way an FYP deepfake
result turns out to be worthless, and `build_manifest.py` is written to
prevent it.

---

## Layout

```
deepfake_system/
  run.sh                     start the server
  app/
    server.py                FastAPI routes and static serving
    engine.py                picks CNN or classical, runs the analysis
    heuristic.py             the classical baseline
    static/index.html
    static/styles.css
    static/app.js
  data/
    build_manifest.py        unify the four datasets, split by identity
    degrade.py               messenger transcode simulator
    dataset.py               clip sampler, paired clean/degraded views
  models/net.py              backbone + SRM stream + temporal attention
  train.py
  evaluate.py                video-level metrics, calibration, breakdown
  infer.py                   CLI scoring for one video
  scripts/make_messenger_testset.sh
  config.py
```

Zero retention is enforced in `app/server.py`: the upload goes to a temp
file and is deleted in a `finally` block. Face thumbnails are returned in
the JSON response and never written to disk. Worth demonstrating live.

---

## Next features worth building

Ranked by marks per hour:

1. **A side-by-side robustness demo** — score a video, transcode it with
   the ffmpeg script, score it again, show both. Demonstrates you
   understand the problem rather than quoting a number.
2. **Batch mode** with CSV export of verdicts.
3. **Model version pinning** in every stored report — checkpoint,
   threshold, calibration. Forensic tools need this.
4. **Grad-CAM** over the highest-scoring frame, shown on click in the
   frame strip. The plumbing is already there.
5. **An audio branch** for lip-sync mismatch. Real work, but it is the
   multimodal requirement, and no competing project will have it.
