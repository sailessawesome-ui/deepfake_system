import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engine import Engine

def main():
    engine = Engine()
    st = engine.status()
    print("=" * 105)
    print("DEEPFAKE VERIFICATION SYSTEM — UPLOADED FOLDERS EVALUATION")
    print(f"Model: {st.get('backbone')} ({st.get('model_version')}) | Calibrated Threshold: {engine.threshold:.3f}")
    print("=" * 105)

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    folders = [
        ("FAKE FOLDER (data/fake)", os.path.join(base, "fake"), 1),
        ("REAL FOLDER (data/real)", os.path.join(base, "real"), 0),
    ]

    tp, fn, tn, fp = 0, 0, 0, 0
    results_list = []

    for title, path, expected in folders:
        files = sorted(glob.glob(os.path.join(path, "*.mp4")))
        print(f"\n>>> {title} — [{len(files)} videos]")
        print(f"{'FILE':42s} | {'VERDICT':13s} | {'SCORE':7s} | {'CONFIDENCE BAND':17s} | {'FACES':5s} | {'RESULT'}")
        print("-" * 105)

        for f in files:
            fname = os.path.basename(f)
            t0 = time.time()
            try:
                res = engine.analyse(f, fname)
                elapsed = time.time() - t0
                verdict = res.get("label", "").upper()
                prob = res.get("probability", 0.0)
                band = res.get("confidence_band", [0.0, 1.0])
                band_str = f"[{band[0]:.3f} - {band[1]:.3f}]"
                faces = res.get("faces_found", 0)

                if expected == 1:
                    is_correct = (verdict == "MANIPULATED") or (verdict == "INCONCLUSIVE" and prob >= engine.threshold)
                    if is_correct:
                        tp += 1
                        tag = "CORRECT (TP)"
                    else:
                        fn += 1
                        tag = "MISS (FN)"
                else:
                    is_correct = (verdict == "AUTHENTIC") or (verdict == "INCONCLUSIVE" and prob < engine.threshold)
                    if is_correct:
                        tn += 1
                        tag = "CORRECT (TN)"
                    else:
                        fp += 1
                        tag = "FALSE ALARM (FP)"

                print(f"{fname[:42]:42s} | {verdict:13s} | {prob:7.4f} | {band_str:17s} | {faces:5d} | {tag} ({elapsed:.1f}s)")
                
                prov = res.get("provenance", {})
                if prov.get("likely_recompressed"):
                    print("   [Provenance] WhatsApp transcode detected (re-compressed, stripped metadata)")
                audio_res = res.get("audio", {})
                if audio_res.get("lipsync", {}).get("reading"):
                    print(f"   [Audio-Visual] Lip-sync agreement: {audio_res['lipsync']['reading']} (score: {audio_res['lipsync'].get('score', 0):.2f})")
                for note in res.get("notes", []):
                    if "Running the classical" not in note:
                        print(f"   [Note] {note}")
                        
                results_list.append({
                    "file": fname, "expected": expected, "verdict": verdict,
                    "prob": prob, "band": band, "faces": faces, "correct": is_correct
                })
            except Exception as exc:
                print(f"{fname[:42]:42s} | ERROR: {exc}")

    n_fake = tp + fn
    n_real = tn + fp
    print("\n" + "=" * 105)
    print("DETECTION & ERROR MATRIX SUMMARY:")
    print("=" * 105)
    print(f"  * FAKE VIDEOS (Ground Truth = FAKE, {n_fake} total):")
    print(f"    - Correctly Detected (TP) : {tp:2d} / {n_fake} ({(tp/max(1,n_fake)):.1%}) [Ratio Fake->Fake: {tp/max(1,n_fake):.2f}]")
    print(f"    - Missed as Real     (FN) : {fn:2d} / {n_fake} ({(fn/max(1,n_fake)):.1%}) [Ratio Fake->Real: {fn/max(1,n_fake):.2f}]")
    print()
    print(f"  * REAL VIDEOS (Ground Truth = REAL, {n_real} total):")
    print(f"    - Correctly Cleared  (TN) : {tn:2d} / {n_real} ({(tn/max(1,n_real)):.1%}) [Ratio Real->Real: {tn/max(1,n_real):.2f}]")
    print(f"    - False Alarms       (FP) : {fp:2d} / {n_real} ({(fp/max(1,n_real)):.1%}) [Ratio Real->Fake: {fp/max(1,n_real):.2f}]")
    print()
    print(f"  * OVERALL ACCURACY          : {(tp+tn)/max(1, n_fake+n_real):.1%} ({tp+tn}/{n_fake+n_real})")
    print("=" * 105)

if __name__ == "__main__":
    main()
