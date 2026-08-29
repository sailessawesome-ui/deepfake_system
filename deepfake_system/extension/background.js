/* Context-menu entry on any video element or page. IR 3.4.1 UI/UX:
   "ideally in the form of a browser extension" (76.8% of respondents),
   compatible with Instagram, Facebook, TikTok and X.

   The extension fetches the media bytes in the page's own context —
   which is what makes it work on sites with authentication/cookies —
   and posts them to the local verification server (http://localhost:8000). */

const DEFAULTS = { server: "http://127.0.0.1:8000" };

function setupContextMenu() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "check-video",
      title: "🔍 Check this video for manipulation (Deepfake Forensics)",
      contexts: ["video", "page", "frame", "link"]
    });
  });
}

chrome.runtime.onInstalled.addListener(setupContextMenu);
chrome.runtime.onStartup.addListener(setupContextMenu);

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "check-video") return;
  const { server } = await chrome.storage.sync.get(DEFAULTS);
  const pageHost = tab?.url ? new URL(tab.url).hostname : "media";
  notify("Deepfake Forensics", `Extracting media from ${pageHost}...`);

  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: grabVideo,
      args: [info.srcUrl || ""]
    });

    if (!result || result.error) {
      throw new Error(result?.error || "No playable video found on this page.");
    }

    notify("Analyzing Evidence", "Running spatial-temporal & multimodal audio verification...");

    const bytes = Uint8Array.from(atob(result.b64), c => c.charCodeAt(0));
    const body = new FormData();
    body.append("video", new Blob([bytes], { type: result.type }), result.name);

    const res = await fetch(`${server}/api/analyse`, { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "The server rejected the file.");

    await chrome.storage.local.set({ lastResult: data });
    const pct = data.probability == null ? "—" : `${Math.round(data.probability * 100)}%`;
    const labelHeader = labelText(data.label);
    notify(`${labelHeader} (${pct})`, `${data.faces_found || 0} faces isolated · Report ready`);

    // Open the Forensic Lab Web App pre-loaded with the full analysis report!
    chrome.tabs.create({ url: `${server}/?view=latest`, active: true });
  } catch (err) {
    notify("Forensic Notice", err.message);
    const noticeEncoded = encodeURIComponent(err.message);
    chrome.tabs.create({ url: `${server}/?notice=${noticeEncoded}&source=${encodeURIComponent(pageHost)}`, active: true });
  }
});

/* Runs inside the page context to handle Facebook / Instagram / X transparent overlays & cookies */
async function grabVideo(passedSrc) {
  try {
    let src = passedSrc;

    // If no direct srcUrl from right-click (e.g. clicked on Facebook overlay div), find video in DOM
    if (!src || src.startsWith("blob:")) {
      const videos = Array.from(document.querySelectorAll("video"));
      if (videos.length === 0) {
        return { error: "No video player detected on this page." };
      }

      // Prioritize currently playing or visible video
      let bestVideo = videos.find(v => !v.paused && v.currentTime > 0) ||
                      videos.find(v => v.offsetWidth > 100 && v.offsetHeight > 100) ||
                      videos[0];

      src = bestVideo.currentSrc || bestVideo.src || "";
      if (!src) {
        const sourceEl = bestVideo.querySelector("source");
        if (sourceEl) src = sourceEl.src;
      }
    }

    if (!src) {
      return { error: "Could not locate the active video stream URL on this page." };
    }

    if (src.startsWith("blob:")) {
      return {
        error: "This platform (e.g. YouTube/FB DASH) uses encrypted media chunks. Download the MP4 clip or save the video, then drop it into the Forensic Lab Web App at http://localhost:8000."
      };
    }

    const r = await fetch(src, { credentials: "include" });
    if (!r.ok) return { error: `The media host returned HTTP ${r.status}.` };

    const buf = await r.arrayBuffer();
    if (buf.byteLength > 250 * 1024 * 1024) return { error: "Video exceeds maximum 250 MB size limit." };

    let bin = "";
    const view = new Uint8Array(buf);
    for (let i = 0; i < view.length; i += 8192) {
      bin += String.fromCharCode.apply(null, view.subarray(i, i + 8192));
    }

    const type = r.headers.get("content-type") || "video/mp4";
    const ext = type.includes("webm") ? "webm" : "mp4";
    return { b64: btoa(bin), type, name: `social-video-${Date.now()}.${ext}` };
  } catch (e) {
    return { error: e.message };
  }
}

function labelText(label) {
  return {
    authentic: "AUTHENTIC MEDIA",
    manipulated: "MANIPULATED (DEEPFAKE)",
    inconclusive: "INCONCLUSIVE — HUMAN AUDIT REQUIRED",
    no_face: "NO FACE DETECTED"
  }[label] || "VERIFICATION COMPLETE";
}

function notify(title, message) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icon128.png",
    title: title || "Deepfake Forensics",
    message: message || ""
  });
}

chrome.notifications.onClicked.addListener(() => {
  chrome.tabs.create({ url: "http://localhost:8000" });
});
