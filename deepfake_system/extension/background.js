/* Context-menu entry on any video element. IR 3.4.1 UI/UX:
   "ideally in the form of a browser extension" (76.8% of respondents),
   compatible with Instagram, Facebook, TikTok and X.

   The extension never analyses anything itself. It fetches the media
   bytes in the page's own context — which is what makes it work on
   sites that require a session — and posts them to the user's own
   server. No third party sees the video. */

const DEFAULTS = { server: "http://localhost:8000" };

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "check-video",
    title: "Check this video for manipulation",
    contexts: ["video"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "check-video" || !info.srcUrl) return;
  const { server } = await chrome.storage.sync.get(DEFAULTS);
  notify("Sending to your server", new URL(info.srcUrl).hostname);

  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: grabVideo,
      args: [info.srcUrl]
    });
    if (!result || result.error) throw new Error(result?.error || "Could not read the video.");

    const bytes = Uint8Array.from(atob(result.b64), c => c.charCodeAt(0));
    const body = new FormData();
    body.append("video", new Blob([bytes], { type: result.type }), result.name);

    const res = await fetch(`${server}/api/analyse`, { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "The server rejected the file.");

    await chrome.storage.local.set({ lastResult: data });
    const pct = data.probability == null ? "—" : `${Math.round(data.probability * 100)}%`;
    notify(labelText(data.label), `${pct} manipulation score · ${data.faces_found} face crops`);
  } catch (err) {
    notify("Check failed", err.message);
  }
});

/* Runs inside the page so cookies and referrer apply. */
async function grabVideo(src) {
  try {
    if (src.startsWith("blob:")) {
      return { error: "This site streams the video in fragments, so it cannot be pulled from the page. Download it first, then upload it on the web interface." };
    }
    const r = await fetch(src, { credentials: "include" });
    if (!r.ok) return { error: `The site returned ${r.status} for that video.` };
    const buf = await r.arrayBuffer();
    if (buf.byteLength > 250 * 1024 * 1024) return { error: "That video is over 250 MB." };
    let bin = "";
    const view = new Uint8Array(buf);
    for (let i = 0; i < view.length; i += 8192) {
      bin += String.fromCharCode.apply(null, view.subarray(i, i + 8192));
    }
    const type = r.headers.get("content-type") || "video/mp4";
    const ext = type.includes("webm") ? "webm" : "mp4";
    return { b64: btoa(bin), type, name: `page-video-${Date.now()}.${ext}` };
  } catch (e) {
    return { error: e.message };
  }
}

function labelText(label) {
  return { authentic: "Authentic", manipulated: "Manipulated",
           inconclusive: "Inconclusive — needs a person",
           no_face: "No face found" }[label] || "Checked";
}

function notify(title, message) {
  chrome.notifications.create({
    type: "basic", iconUrl: "icon128.png", title, message: message || ""
  });
}

chrome.notifications.onClicked.addListener(() => {
  chrome.tabs.create({ url: "http://localhost:8000" });
});
