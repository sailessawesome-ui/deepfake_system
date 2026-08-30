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
| **Ongoing algorithmic adaptation** | **FIXED** | `app/modelreg.py` — validated hot-swap, no restart |

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
| **Simple explainable output with visual cues (XAI)** | **FIXED** | `app/explain.py` — Grad-CAM |
| "A definite Real or Fake tag" | **Deliberate deviation** | see below |

### Explainable AI

This row previously read "Met — frame strip, margin gauge, signals
panel". That was an overstatement, and worth being honest about: those
show *how confident* the model is and *when* across the clip, but the IR
asks for a cue that explains **why** — "pointing to a particular artifact
on the face". Nothing in the system did that. It was also the single
most-requested feature in the survey, a top priority for more than half of
the 56 respondents.

`app/explain.py` implements **Grad-CAM** (Selvaraju et al., 2017): the
gradient of the model's own frame logit with respect to the last
convolutional feature map, channel-weighted and rectified into a heatmap
over the face crop. Three points to have ready:

- **It explains this model, not deepfakes in general.** The map shows
  where the network looked. If it lands on the background, that is
  evidence the score is being driven by something other than the face —
  which is information the analyst needs, not a bug to hide.
- **It is not a segmentation of the manipulated region.** The response
  ships with that caveat attached, because a heatmap invites over-reading.
- **Only the three most suspicious frames are explained.** Each costs a
  forward and a backward pass, and this trades directly against the
  "near real-time" non-functional requirement. Measured at **2.4 s** on
  CPU for three frames — after an optimisation: the per-frame logit
  depends only on that frame's features, so the explanation pass runs on
  a single frame rather than pushing the whole clip through the backbone.
  The naive version cost 5.5 s for identical output.

In the interface, an explained crop is marked `XAI` and clicking it
toggles between the crop and the heatmap. The plain-English summary
("the strongest response is in the mid-face") is written into the strip
caption, so the cue is readable without technical knowledge — which is
what the requirement actually asks for. The narrative is stored in
`dfd_analyses`; the heatmap images never are.

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
| **Encrypted transport between UI and backend** | **FIXED** | `deploy/nginx.conf` + `app/security.py` — see the Secure processing section |

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
falls back to a local JSON file otherwise — so it runs on your laptop and
on AWS with no code change. `/api/status` reports which backend is live,
so you can demonstrate both.

Do not commit AWS keys. Scope the IAM policy to `deepfake_*` tables and
their `/index/*` ARNs. Provisioning is `python scripts/create_tables.py`;
the full deployment runbook is `DEPLOY_RAILWAY.md`.

### 2.4.4 in full — all three clauses

Section 2.4.4 says DynamoDB stores "detection results, **user metadata,
and system logs**". Only the first was implemented; the other two are now
present, which is also what made login/logout possible.

Live in account `454229054677`, region `ap-southeast-5`. Four of the five
already existed and hold real records, so the code was written to match
their schemas rather than the reverse — see `app/tables.py`.

| Table | Key | Clause of 2.4.4 |
|---|---|---|
| `dfd_analyses` | `user_id` + `analysis_id` | detection results (Table 16) |
| `dfd_users` | `user_id`, GSI `EmailIndex` | user metadata |
| `dfd_sessions` | `token_hash`, GSI `UserSessionsIndex`, TTL `expires_at` | login/logout state |
| `dfd_login_attempts` | `identifier` + `attempted_at`, TTL `ttl` | lockout |
| `dfd_audit_log` | `audit_day` + `event_ts`, GSI `UserAuditIndex`, TTL `expires_at` | system logs |

Two schema decisions were forced by the existing rows and are worth
stating rather than defending as preferences:

- **`user_id` is a random UUIDv4, looked up by the `EmailIndex` GSI.**
  Deriving it from the email would have made uniqueness atomic, but it
  would also have orphaned all 35 existing accounts. The consequence is
  that email uniqueness is a read-then-write check and therefore racy;
  DynamoDB offers no atomic uniqueness on a non-key attribute.
- **Failed logins live in their own table**, not as a counter on the user
  row, so an attempt against an address with no account is still recorded
  and the TTL expires the lockout without a `locked_until` comparison.

**Table 16 deviations, stated deliberately.** The IR lists ten attributes;
the implementation stores eight of them and adds several more.

| Table 16 | Implemented | Why |
|---|---|---|
| `detection_id` (PK) | `report_id` + `created_at` | composite key; `created_at` sorts the GSI |
| `user_id` | `user_id` | **FIXED** — was absent before accounts existed |
| `video_filename` | `filename` | — |
| `video_s3_uri` | *omitted* | zero retention (IR 3.4.1): no media is stored |
| `detection_status` | *omitted* | analysis is synchronous; no pending state exists |
| `confidence_score` | `probability` | stored as `Decimal`; DynamoDB rejects float |
| `classification` | `label` + `confidence_band` | — |
| `frame_results` | *omitted* | `frames` is stripped in `_STRIP`; `clips_scored` kept |
| `processing_timestamp` | `created_at` | ISO-8601, because it doubles as the sort key |
| `model_version` | `model_version` | — |

Either amend Table 16 in the report or be ready to explain the three
omissions. All three follow from requirements stated elsewhere in the IR,
so the deviation is defensible — but only if you raise it first.

---

## Chapter 3.4.1 — Ongoing algorithmic adaptation (non-functional 4)

"The system architecture should enable the ongoing updating of its machine
learning models." Previously this meant overwriting `runs/v1/best.pt` and
restarting: the service went down, a bad checkpoint was only discovered
once it was already serving, and nothing recorded that the model changed.

`app/modelreg.py` plus three admin-only endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/models` | every checkpoint under `runs/`, with its training metrics |
| `POST /api/models/validate` | dry run — loads and scores without going live |
| `POST /api/models/activate` | validates, then swaps in without a restart |

Four properties worth defending:

- **Validated before activation.** A candidate is built as a *separate*
  model and made to score a probe clip. Truncated files, architecture
  mismatches and checkpoints that emit NaN are all rejected while the
  running model keeps serving. A NaN-producing checkpoint passes every
  structural check and then poisons every verdict, which is exactly why
  the smoke test exists.
- **Atomic.** The swap happens under a lock, so an analysis already in
  flight finishes on the model it started with.
- **Admin-only.** The detection model is what the whole forensic claim
  rests on. If any account could change it, no report would be
  falsifiable — you could never say which weights produced a verdict or
  who chose them. Gated on `is_admin` in `dfd_users`.
- **Audited.** `model.activated` and `model.rejected` both go to
  `dfd_audit_log`, and `model_version` is already on every stored
  analysis, so any past verdict traces to the weights behind it.

Checkpoint ids are path-checked against `runs/`, so `../../etc/passwd`
is refused rather than resolved. Listing uses `weights_only=True` — a
`.pt` file is a pickle, and enumerating models must not be a way to
execute code.

---

## Chapter 3.4.1 — Secure processing (security requirement 2)

"The data sent between the user interface and the backend processing
server should be heavily encrypted to avoid interception or
manipulation." Everything was plain HTTP, and the nginx config that once
handled TLS had been removed from the repo. This was genuinely unmet.

| Layer | Where |
|---|---|
| TLS 1.2/1.3 termination, HSTS, OCSP stapling | `deploy/nginx.conf` |
| Security headers, CSP, https redirect | `app/security.py` |
| Hardened systemd unit | `deploy/deepfake.service` |
| Local HTTPS for testing and demos | `scripts/dev_tls.py` |

**The CSP is the part worth pointing at.** The interface has no inline
`<script>` blocks at all, so `script-src` is `'self'` with no
`unsafe-inline` and no `unsafe-eval`: an injected `<script>` tag does not
execute. That is the control that would have *contained* the stored-XSS
hole found in the account badge, rather than relying on output escaping
alone — defence in depth, with both layers now present.

`style-src` does carry `'unsafe-inline'`, because the page and the report
sheet use a handful of `style=""` attributes. Stated rather than hidden;
removing it is a small refactor if an examiner presses.

Verified locally over real TLS (`python scripts/dev_tls.py`): TLS 1.3,
`TLS_AES_256_GCM_SHA384`, HSTS present over https and correctly absent
over http, and `Cache-Control: no-store` on every `/api/` response so an
evidence report is never held in a shared cache.

---

## Chapter 3.4.1 — Authentication

The sign-in UI existed but was theatre: `localStorage`, plaintext
passwords in the browser, `isLoggedIn()` hardcoded to `return true`, and
an unrecognised email silently auto-registered with whatever password was
typed. Anyone could reach `/api/analyse` with curl regardless.

| Requirement | Status | Where |
|---|---|---|
| Restrict ingestion to authenticated investigators | **FIXED** | `require_user` in `app/server.py` |
| Password storage | **FIXED** | PBKDF2-HMAC-SHA256, 600k rounds, `app/auth.py` |
| Session management | **FIXED** | `deepfake_sessions`, HttpOnly + Secure + SameSite=Lax cookie |
| Chain-of-custody logging (ISO/IEC 27037) | **FIXED** | `app/audit.py` |
| Per-analyst evidence scoping | **FIXED** | `store.for_user()`, GSI query |

Three vulnerabilities were found and closed while wiring this up, all of
them created by the move from one implicit operator to real accounts:

- **`_LATEST_RESULT` was a single module global.** `/api/reports/latest`
  returned whatever the *previous caller* had analysed, whoever they were.
  Now keyed by user.
- **The header badge rendered `user.name` through `innerHTML`.** With
  names coming from a database, registering as `<img src=x onerror=...>`
  is stored XSS against every viewer. Now escaped.
- **`allow_origins=["*"]` with `allow_credentials=True`.** Invalid per the
  CORS spec; browsers drop the response. It would have broken session
  cookies on the first cross-origin call. Now an explicit origin list plus
  a regex for the extension.

---

## Still open

Two things the report asks for that are **not** done, listed so you are
not surprised by them:

1. ~~**A trained checkpoint.**~~ Done — `runs/v1/best.pt`,
   `tf_efficientnetv2_s`, val F1 0.9821. One caveat for deployment:
   `runs/` is gitignored and the file is 81 MB, so a git-based deploy does
   not carry it. `app/modelfetch.py` fetches it from S3 at boot; without
   `DF_MODEL_S3_URI` set, the hosted app silently reverts to the classical
   baseline and its numbers will not match Chapter 4.
2. **A mobile app.** The UI/UX requirement says "browser extension **or**
   a feature of a mobile application". The extension satisfies the
   disjunction. The web interface is responsive, so a mobile browser
   works, but there is no native app and you should not claim one.

---

## New API surface

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /api/status` | optional | engine, model version, branches; table detail only when signed in |
| `POST /api/auth/register` | — | create an account, sets the session cookie |
| `POST /api/auth/login` | — | sign in |
| `POST /api/auth/logout` | — | revoke this session |
| `POST /api/auth/logout-all` | required | revoke every session for the account |
| `GET /api/auth/me` | optional | who is signed in, or `{"user": null}` |
| `POST /api/analyse` | **required** | returns `audio`, `content_credentials`, `timings`, `model_version`, `report_id`, `seen_before` |
| `POST /api/analyse-url` | **required** | same, from a stream URL |
| `GET /api/reports?limit=25` | required | the caller's own past findings |
| `GET /api/reports/by-hash/{sha256}` | required | the caller's reports for one file |
| `GET /api/audit?limit=50` | required | the caller's own activity trail |

`/api/analyse` and `/api/analyse-url` now return **401** to an anonymous
caller. That is the one breaking change: any script calling them must
sign in first and keep the cookie, or send `Authorization: Bearer <token>`
using the token returned by `/api/auth/login`. Set `DF_REQUIRE_LOGIN=false`
to restore open access for a local demo.
