/* Context-menu entry on any video element or page. IR 3.4.1 UI/UX:
   "ideally in the form of a browser extension" (76.8% of respondents),
   compatible with Instagram, Facebook, TikTok and X.

   Seamlessly handles direct video containers AND encrypted DASH streaming
   platforms (YouTube, Facebook, Instagram, TikTok, X) via server-side
   ingestion (yt-dlp) and extension background workers. */

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
  notify("Deepfake Forensics", `Ingesting video from ${pageHost}...`);

  try {
    // 1. Locate video URL or determine platform streaming type
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: locateVideoOnPage,
      args: [info.srcUrl || "", tab.url || ""]
    });

    if (!result || result.error) {
      throw new Error(result?.error || "No active video stream located on this page.");
    }

    let data;

    // 2A. Social / DASH platforms (YouTube, Facebook, TikTok, Instagram, X): Open Lab instantly with streaming URL
    if (result.useUrlEngine) {
      notify("Ingesting Stream", `Opening Forensic Lab for ${result.domain || pageHost}...`);
      chrome.tabs.create({ url: `${server}/?url=${encodeURIComponent(result.pageUrl)}`, active: true });
      return;
    } else {
      // 2B. Direct Media File: Fetch buffer in Extension Worker
      notify("Downloading Evidence", `Extracting media buffer from ${result.domain || pageHost}...`);
      const mediaRes = await fetch(result.url, { mode: "cors" }).catch(() => fetch(result.url));
      if (!mediaRes.ok) {
        throw new Error(`Media CDN returned HTTP ${mediaRes.status}. The video URL may be expired or DRM protected.`);
      }

      const blob = await mediaRes.blob();
      if (blob.size < 1000) {
        throw new Error("Extracted video stream is empty or corrupt.");
      }
      if (blob.size > 250 * 1024 * 1024) {
        throw new Error("Video exceeds maximum 250 MB size limit.");
      }

      notify("Analyzing Media", "Running spatial-temporal & multimodal audio verification...");

      const filename = `media_${Date.now()}.${result.ext || "mp4"}`;
      const body = new FormData();
      body.append("video", blob, filename);

      const res = await fetch(`${server}/api/analyse`, { method: "POST", body });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "The verification engine rejected the media file.");
    }

    await chrome.storage.local.set({ lastResult: data });
    const pct = data.probability == null ? "—" : `${Math.round(data.probability * 100)}%`;
    const labelHeader = labelText(data.label);
    notify(`${labelHeader} (${pct})`, `${data.faces_found || 0} faces isolated · Full report ready`);

    // Open the Forensic Lab Web App pre-loaded with the full analysis report!
    chrome.tabs.create({ url: `${server}/?view=latest`, active: true });
  } catch (err) {
    notify("Forensic Notice", err.message);
    const noticeEncoded = encodeURIComponent(err.message);
    chrome.tabs.create({ url: `${server}/?notice=${noticeEncoded}&source=${encodeURIComponent(pageHost)}`, active: true });
  }
});

/* Inspects DOM to locate video URL or flag social media DASH platforms */
function locateVideoOnPage(passedSrc, currentTabUrl) {
  try {
    let src = passedSrc;
    const pageUrl = window.location.href || currentTabUrl;
    const domain = window.location.hostname || "media";

    const isSocialHost = /youtube\.com|youtu\.be|facebook\.com|instagram\.com|tiktok\.com|twitter\.com|x\.com/i.test(pageUrl);

    // If on YouTube, Facebook, Instagram, TikTok, X, route directly to URL Engine for perfect stream extraction!
    if (isSocialHost) {
      return { useUrlEngine: true, pageUrl, domain };
    }

    // Otherwise check for DOM video element
    if (!src || src.startsWith("blob:")) {
      const videos = Array.from(document.querySelectorAll("video"));
      if (videos.length === 0) {
        return { error: "No video player detected on this page." };
      }

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
      return { error: "Could not locate active video stream URL on this page." };
    }

    if (src.startsWith("blob:")) {
      return { useUrlEngine: true, pageUrl, domain };
    }

    let ext = "mp4";
    try {
      const u = new URL(src);
      if (u.pathname.endsWith(".webm")) ext = "webm";
      if (u.pathname.endsWith(".mov")) ext = "mov";
    } catch (_) {}

    return { useUrlEngine: false, url: src, ext, domain, pageUrl };
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
  chrome.tabs.create({ url: "http://127.0.0.1:8000" });
});
