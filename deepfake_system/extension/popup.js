document.getElementById("opts").addEventListener("click", e => {
  e.preventDefault(); chrome.runtime.openOptionsPage();
});
chrome.storage.local.get("lastResult", ({ lastResult: d }) => {
  if (!d) return;
  const cls = { authentic: "real", manipulated: "fake" }[d.label] || "maybe";
  const word = { authentic: "Authentic", manipulated: "Manipulated",
                 inconclusive: "Inconclusive", no_face: "No face found" }[d.label] || d.label;
  const pct = d.probability == null ? "—" : d.probability.toFixed(3);
  const band = d.confidence_band ? `margin ${d.confidence_band[0].toFixed(2)}–${d.confidence_band[1].toFixed(2)}` : "";
  document.getElementById("out").innerHTML =
    `<p class="v ${cls}">${word}</p><p class="n">${pct} · ${band}</p>` +
    `<p>${(d.notes && d.notes[0]) || ""}</p>`;
});
