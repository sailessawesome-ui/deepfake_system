const input = document.getElementById("server");
const msg = document.getElementById("msg");
chrome.storage.sync.get({ server: "http://localhost:8000" }, s => input.value = s.server);
document.getElementById("save").addEventListener("click", async () => {
  const server = input.value.trim().replace(/\/+$/, "");
  if (!/^https?:\/\/.+/.test(server)) { msg.style.color = "#F2557E"; msg.textContent = "That does not look like a server address."; return; }
  await chrome.storage.sync.set({ server });
  msg.style.color = "#34C8A5";
  try {
    const r = await fetch(`${server}/api/status`);
    const d = await r.json();
    msg.textContent = `Saved. Server is running the ${d.mode} engine.`;
  } catch { msg.textContent = "Saved, but the server did not answer. Is it running?"; }
});
