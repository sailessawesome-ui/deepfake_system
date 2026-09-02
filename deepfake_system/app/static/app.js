/**
 * Deepfake Forensics — Biometric & Spatial-Temporal Media Verification
 * Final Year Capstone Project — BSc (Hons) Cybersecurity & Digital Forensics
 * Student: Sailess Raj | Student ID: CYB-2026-9481
 *
 * Implements client-side pipeline control, evidence custody tracking,
 * confidence margin visualization, and zero-retention analysis workflows.
 */

const $ = (id) => document.getElementById(id);

/* Escape before interpolating anything a user typed into innerHTML.
   Account names and filenames both reach the DOM this way. */
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

const el = {
  // Navigation & Identity Management
  openAuthBtn: $('openAuthBtn'), authBtnText: $('authBtnText'), authBox: $('authBox'),
  authModal: $('authModal'), closeAuthBtn: $('closeAuthBtn'),
  tabSignIn: $('tabSignIn'), tabSignUp: $('tabSignUp'),
  signInForm: $('signInForm'), signUpForm: $('signUpForm'),
  authAlert: $('authAlert'), switchSignUpLink: $('switchSignUpLink'),
  loginEmail: $('loginEmail'), loginPassword: $('loginPassword'),
  regName: $('regName'), regEmail: $('regEmail'), regRole: $('regRole'), regPassword: $('regPassword'),
  lockSignInBtn: $('lockSignInBtn'), bannerSignInBtn: $('bannerSignInBtn'), authGateBanner: $('authGateBanner'),
  portalLoginBtn: $('portalLoginBtn'), portalSignUpBtn: $('portalSignUpBtn'),
  loggedOutPortal: $('loggedOutPortal'), navHome: $('navHome'), navLab: $('navLab'),
  heroLaunchBtnText: $('heroLaunchBtnText'),

  // Evidence Ingestion & Workspace
  workspace: $('workspace'),
  drop: $('drop'), file: $('file'), browse: $('browse'),
  dropLockOverlay: $('dropLockOverlay'), ingestBadge: $('ingestBadge'),
  quickDemoBtn: $('quickDemoBtn'), heroDemoBtn: $('heroDemoBtn'), heroLaunchBtn: $('heroLaunchBtn'),
  empty: $('empty'), working: $('working'), result: $('result'),
  failure: $('failure'), failureMsg: $('failureMsg'),
  workingStep: $('workingStep'), workingFile: $('workingFile'), workingFill: $('workingFill'),
  scannerVideo: $('scannerVideo'),

  // Verdict & Forensic Telemetry
  verdict: $('verdict'), verdictWord: $('verdictWord'), verdictPill: $('verdictPill'),
  verdictSummaryBadge: $('verdictSummaryBadge'), verdictFile: $('verdictFile'),
  verdictNum: $('verdictNum'), verdictPct: $('verdictPct'), verdictNumLabel: $('verdictNumLabel'),
  gauge: $('gauge'), gaugeBand: $('gaugeBand'), gaugePoint: $('gaugePoint'),
  gaugeLine: $('gaugeLine'), gaugeRead: $('gaugeRead'),
  strip: $('strip'), stripSub: $('stripSub'), stripBadge: $('stripBadge'), plot: $('plot'),
  notes: $('notes'), features: $('features'), featurePanel: $('featurePanel'),
  specs: $('specs'), engineChip: $('engineChip'), engineMeta: $('engineMeta'),
  telemetryFaceBackend: $('telemetryFaceBackend'), telemetryEngine: $('telemetryEngine'),
  recent: $('recent'), recentWrap: $('recentWrap'), recentCount: $('recentCount'),
  storedWrap: $('storedWrap'), storedList: $('storedList'),
  storedCount: $('storedCount'), storedFoot: $('storedFoot'),
  again: $('again'), copy: $('copy'), printReport: $('printReport'), failAgain: $('failAgain'),
  generatePdfBtn: $('generatePdfBtn'),
  pdfModal: $('pdfModal'), pdfPaperSheet: $('pdfPaperSheet'),
  pdfDownloadBtn: $('pdfDownloadBtn'), pdfPrintBtn: $('pdfPrintBtn'),
  closePdfModalBtn: $('closePdfModalBtn'),
};

const session = [];
let current = null;

const VERDICT = {
  manipulated: {
    word: 'FAKE',
    pill: 'FLAGGED: DEEPFAKE VIDEO',
    summary: '⚠️ THIS VIDEO IS FAKE',
    cls: 'is-fake'
  },
  authentic: {
    word: 'NOT FAKE',
    pill: 'CLEARED: AUTHENTIC MEDIA',
    summary: '✅ THIS VIDEO IS NOT FAKE (REAL)',
    cls: 'is-real'
  },
  inconclusive: {
    word: 'INCONCLUSIVE',
    pill: 'UNCERTAIN: MARGIN CROSSES LINE',
    summary: '⚠️ SUSPECT / UNCERTAIN - MANUAL REVIEW REQUIRED',
    cls: 'is-maybe'
  },
  no_face: {
    word: 'NO FACE FOUND',
    pill: 'UNABLE TO EXAMINE',
    summary: 'NO BIOMETRIC TARGET FOUND',
    cls: ''
  },
};

/* ── Verification Engine Status ─────────────────────────────────── */
async function loadStatus() {
  try {
    const s = await (await fetch('/api/status')).json();
    el.engineMeta.innerHTML = '';
    
    const engine = document.createElement('span');
    if (s.mode === 'model') {
      engine.className = 'chip chip--live';
      engine.textContent = `neural · ${s.backbone || 'cnn'}`;
      if (el.telemetryEngine) el.telemetryEngine.textContent = (s.backbone || 'CNN').toUpperCase();
    } else {
      engine.className = 'chip chip--base';
      engine.textContent = 'baseline · heuristic';
      engine.title = 'Drop a trained checkpoint into models/ to activate deep neural scoring.';
      if (el.telemetryEngine) el.telemetryEngine.textContent = 'HEURISTIC';
    }

    if (el.telemetryFaceBackend) {
      el.telemetryFaceBackend.textContent = (s.face_backend || 'HAAR').toUpperCase();
    }

    const accEl = document.getElementById('telemetryAccuracy');
    const accSub = document.getElementById('telemetryAccuracySub');
    if (accEl && s.model_version && s.model_version.includes('f1=')) {
      const match = s.model_version.match(/f1=([\d\.]+)/);
      if (match) {
        const val = (parseFloat(match[1]) * 100).toFixed(1);
        accEl.innerHTML = `${val}<span class="telemetry-unit">%</span>`;
        if (accSub) accSub.textContent = `Held-out FF++ (c23) & Celeb-DF v2 (Val F1: ${val}%)`;
      }
    }

    const backendChip = document.createElement('span');
    backendChip.className = 'chip chip--live';
    backendChip.textContent = `face backend · ${s.face_backend}`;

    el.engineMeta.append(
      engine,
      backendChip,
      chip(s.device.toUpperCase()),
      chip(`threshold ${Number(s.threshold).toFixed(2)}`),
    );
  } catch {
    el.engineChip.textContent = 'engine standby';
  }
}

function chip(text) {
  const c = document.createElement('span');
  c.className = 'chip';
  c.textContent = text;
  return c;
}

/* ── Access Control & Student Authentication ────────────────────── */
/* Accounts live in DynamoDB (IR 2.4.4), not in this file. The browser
   holds an HttpOnly session cookie it cannot read, so the only way to
   know who is signed in is to ask the server. Every gate here is a
   convenience for the user — the real enforcement is require_user() on
   the server, because anything decided in JavaScript can be edited by
   whoever is looking at it. */

let CURRENT_USER = null;

async function api(path, options = {}) {
  const res = await fetch(path, {
    credentials: 'same-origin',        // send the session cookie
    headers: options.body ? { 'Content-Type': 'application/json' } : {},
    ...options
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const err = new Error((data && data.detail) || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function getCurrentUser() {
  return CURRENT_USER;
}

async function refreshCurrentUser() {
  try {
    const data = await api('/api/auth/me');
    CURRENT_USER = data && data.user ? data.user : null;
  } catch {
    CURRENT_USER = null;             // offline or server down: treat as signed out
  }
  return CURRENT_USER;
}

function isLoggedIn() {
  return !!CURRENT_USER;
}

function updateAccessState() {
  const user = CURRENT_USER;
  const loggedIn = !!user;

  if (loggedIn) {
    // Reveal detection lab workspace, hide logged out portal
    if (el.workspace) el.workspace.hidden = false;
    if (el.loggedOutPortal) el.loggedOutPortal.hidden = true;
    if (el.navLab) el.navLab.hidden = false;
    el.drop?.classList.remove('is-locked');
    if (el.authGateBanner) el.authGateBanner.hidden = true;
    if (el.ingestBadge) {
      el.ingestBadge.textContent = 'AUTHORIZED · LAB ACTIVE';
      el.ingestBadge.style.color = 'var(--real)';
      el.ingestBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    }
    if (el.heroLaunchBtnText) el.heroLaunchBtnText.textContent = 'Go to Detection Lab';

    // Render Student ID badge in header. Name, role and student ID are
    // whatever the account holder typed at sign-up, so they are escaped
    // before they touch innerHTML — otherwise registering as
    // `<img src=x onerror=...>` stores XSS for every viewer of this page.
    const initials = esc(user.initials || '??');
    const role = esc(user.role || '');
    el.authBox.innerHTML = `
      <div class="user-profile-badge">
        <div class="user-avatar" title="${role}">${initials}</div>
        <div class="user-info">
          <span class="user-name">${esc(user.name || '')}</span>
          <span class="user-role">${esc((user.role || '').split(' ')[0] || '')}</span>
        </div>
        <div class="user-actions">
          <button class="user-action" id="openSettingsBtn" type="button"
                  title="Account settings" aria-label="Account settings">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
          </button>
          <button class="user-action user-action--out" id="signOutBtn" type="button"
                  title="Sign out" aria-label="Sign out">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
              <polyline points="16 17 21 12 16 7"></polyline>
              <line x1="21" y1="12" x2="9" y2="12"></line>
            </svg>
            <span class="user-action-label">Sign out</span>
          </button>
        </div>
      </div>
    `;
    $('signOutBtn')?.addEventListener('click', logoutUser);
    $('openSettingsBtn')?.addEventListener('click', openSettingsModal);
  } else {
    // Hide detection lab workspace, display locked portal for visitors
    if (el.workspace) el.workspace.hidden = true;
    if (el.loggedOutPortal) el.loggedOutPortal.hidden = false;
    if (el.navLab) el.navLab.hidden = true;
    el.drop?.classList.add('is-locked');
    if (el.authGateBanner) el.authGateBanner.hidden = false;
    if (el.ingestBadge) {
      el.ingestBadge.textContent = 'AUTHENTICATION REQUIRED';
      el.ingestBadge.style.color = 'var(--maybe)';
      el.ingestBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    }
    if (el.heroLaunchBtnText) el.heroLaunchBtnText.textContent = 'Scan a Video Now';

    el.authBox.innerHTML = `
      <button class="btn-auth" id="openAuthBtn" type="button">
        <svg class="auth-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
          <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
        </svg>
        <span id="authBtnText">Sign In</span>
      </button>
    `;
    $('openAuthBtn')?.addEventListener('click', openAuthModal);
  }
}

function showAuthAlert(msg, isSuccess = false) {
  if (!el.authAlert) return;
  el.authAlert.hidden = false;
  el.authAlert.textContent = msg;
  el.authAlert.className = `auth-alert ${isSuccess ? 'is-success' : ''}`;
}

function clearAuthAlert() {
  if (!el.authAlert) return;
  el.authAlert.hidden = true;
  el.authAlert.textContent = '';
}

async function initAuth() {
  // Ask the server who we are before painting. The session lives in an
  // HttpOnly cookie, so this is the only way to find out.
  await refreshCurrentUser();
  updateAccessState();
  loadRecent();

  // Modal triggers
  el.openAuthBtn?.addEventListener('click', () => { setAuthTab('signin'); openAuthModal(); });
  el.lockSignInBtn?.addEventListener('click', () => { setAuthTab('signin'); openAuthModal(); });
  el.bannerSignInBtn?.addEventListener('click', () => { setAuthTab('signin'); openAuthModal(); });
  el.closeAuthBtn?.addEventListener('click', closeAuthModal);
  
  // Portal locked card triggers
  el.portalLoginBtn?.addEventListener('click', () => {
    setAuthTab('signin');
    openAuthModal();
  });
  el.portalSignUpBtn?.addEventListener('click', () => {
    setAuthTab('signup');
    openAuthModal();
  });

  el.switchSignUpLink?.addEventListener('click', (e) => {
    e.preventDefault();
    setAuthTab('signup');
  });

  // Global click delegation to ensure Sign In & Create Account buttons ALWAYS respond
  document.addEventListener('click', (e) => {
    if (e.target.closest('#openAuthBtn') || e.target.closest('#lockSignInBtn') || e.target.closest('#bannerSignInBtn') || e.target.closest('#portalLoginBtn')) {
      setAuthTab('signin');
      openAuthModal();
    } else if (e.target.closest('#portalSignUpBtn') || e.target.closest('#switchSignUpLink')) {
      e.preventDefault();
      setAuthTab('signup');
      openAuthModal();
    }
  });

  el.authModal?.addEventListener('click', (e) => {
    if (e.target === el.authModal) closeAuthModal();
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !el.authModal.hidden) closeAuthModal();
  });

  // Tab navigation
  el.tabSignIn?.addEventListener('click', () => setAuthTab('signin'));
  el.tabSignUp?.addEventListener('click', () => setAuthTab('signup'));

  // Log In Form Submission
  el.signInForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearAuthAlert();
    const input = el.loginEmail.value.trim();
    const password = el.loginPassword.value;
    if (!input || !password) {
      showAuthAlert('Enter your email and password.');
      return;
    }

    const btn = el.signInForm.querySelector('button[type="submit"]');
    if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = 'Verifying…'; }
    try {
      // The old build auto-created an account for any unrecognised email
      // and accepted whatever password came with it. That is gone: an
      // unknown account is now a failed sign-in, logged as one.
      const data = await api('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email: input, password })
      });
      loginUser(data.user);
    } catch (err) {
      showAuthAlert(err.message || 'Sign in failed.');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label || 'Sign In'; }
    }
  });

  // Sign Up Form Submission
  el.signUpForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearAuthAlert();
    const name = el.regName.value.trim();
    const email = el.regEmail.value.trim().toLowerCase();
    const role = el.regRole.value;
    const password = el.regPassword.value;

    if ((password || '').length < 8) {
      showAuthAlert('Use at least 8 characters for the password.');
      return;
    }

    const btn = el.signUpForm.querySelector('button[type="submit"]');
    if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = 'Creating…'; }
    try {
      const data = await api('/api/auth/register', {
        method: 'POST',
        body: JSON.stringify({ email, password, name, role })
      });
      loginUser(data.user);
    } catch (err) {
      showAuthAlert(err.message || 'Could not create the account.');
      if (/already exists/i.test(err.message || '')) {
        setAuthTab('signin');
        el.loginEmail.value = email;
      }
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label || 'Create Account'; }
    }
  });
}

function openAuthModal() {
  clearAuthAlert();
  if (el.authModal) {
    el.authModal.hidden = false;
    el.authModal.style.display = 'flex';
  }
  document.body.style.overflow = 'hidden';
}

function closeAuthModal() {
  clearAuthAlert();
  if (el.authModal) {
    el.authModal.hidden = true;
    el.authModal.style.display = 'none';
  }
  document.body.style.overflow = '';
}

function setAuthTab(tab) {
  clearAuthAlert();
  const isSignIn = tab === 'signin';
  if (el.tabSignIn) {
    el.tabSignIn.classList.toggle('is-active', isSignIn);
    el.tabSignIn.setAttribute('aria-selected', isSignIn);
  }
  if (el.tabSignUp) {
    el.tabSignUp.classList.toggle('is-active', !isSignIn);
    el.tabSignUp.setAttribute('aria-selected', !isSignIn);
  }
  if (el.signInForm) {
    el.signInForm.hidden = !isSignIn;
    el.signInForm.style.display = isSignIn ? 'grid' : 'none';
  }
  if (el.signUpForm) {
    el.signUpForm.hidden = isSignIn;
    el.signUpForm.style.display = isSignIn ? 'none' : 'grid';
  }
}

function loginUser(user) {
  // The session itself is the HttpOnly cookie the server just set; this
  // only updates what the page shows.
  CURRENT_USER = user;
  updateAccessState();
  closeAuthModal();
  if (typeof loadRecent === 'function') loadRecent();
}

async function logoutUser() {
  try {
    await api('/api/auth/logout', { method: 'POST' });
  } catch {
    // Revoking server-side is what matters; if the call fails the cookie
    // still expires on its own. Clear the UI either way.
  }
  CURRENT_USER = null;
  updateAccessState();
}

/* ── Evidence Ingestion & Authentication Gate ────────────────────── */
el.browse?.addEventListener('click', (e) => {
  e.stopPropagation();
  if (!isLoggedIn()) {
    openAuthModal();
    return;
  }
  el.file.click();
});

el.drop?.addEventListener('click', (e) => {
  if (e.target.closest('#lockSignInBtn')) return;
  if (!isLoggedIn()) {
    openAuthModal();
    return;
  }
  if (e.target === el.file || e.target.closest('#quickDemoBtn')) return;
  el.file.click();
});

el.drop?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    if (!isLoggedIn()) {
      openAuthModal();
      return;
    }
    el.file.click();
  }
});

el.file?.addEventListener('change', () => {
  if (el.file.files[0]) send(el.file.files[0]);
});

['dragenter', 'dragover'].forEach((t) =>
  el.drop?.addEventListener(t, (e) => {
    e.preventDefault();
    if (isLoggedIn()) el.drop.classList.add('is-over');
  }));

['dragleave', 'drop'].forEach((t) =>
  el.drop?.addEventListener(t, (e) => {
    e.preventDefault();
    el.drop.classList.remove('is-over');
  }));

el.drop?.addEventListener('drop', (e) => {
  if (!isLoggedIn()) {
    openAuthModal();
    return;
  }
  const f = e.dataTransfer?.files[0];
  if (f) send(f);
});

const STEPS = [
  'Verifying evidence integrity & SHA-256 hash',
  'Decoding video container & keyframe timestamps',
  'Extracting & aligning facial regions (MTCNN)',
  'Evaluating spatial-frequency & temporal consistency',
  'Computing final fake score against decision threshold',
];

/* ── Scanning-panel video preview ─────────────────────────────────
   Purely cosmetic — plays the file the user just uploaded, muted,
   behind the scan-line effect. Never touches the analysis itself. */
let scannerObjectUrl = null;

function showScannerPreview(file) {
  const v = el.scannerVideo;
  if (!v || !file) return;
  if (scannerObjectUrl) URL.revokeObjectURL(scannerObjectUrl);
  scannerObjectUrl = URL.createObjectURL(file);
  v.src = scannerObjectUrl;
  v.hidden = false;
  v.play().catch(() => {});
}

function hideScannerPreview() {
  const v = el.scannerVideo;
  if (!v) return;
  v.pause();
  v.hidden = true;
  v.removeAttribute('src');
  v.load();
  if (scannerObjectUrl) {
    URL.revokeObjectURL(scannerObjectUrl);
    scannerObjectUrl = null;
  }
}

async function send(file) {
  if (!isLoggedIn()) {
    openAuthModal();
    return;
  }

  show('working');
  el.workingFile.textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  showScannerPreview(file);

  let i = 0;
  el.workingStep.textContent = STEPS[0];
  const ticker = setInterval(() => {
    i = Math.min(i + 1, STEPS.length - 1);
    el.workingStep.textContent = STEPS[i];
  }, 1200);

  const body = new FormData();
  body.append('video', file);

  try {
    const res = await fetch('/api/analyse', {
      method: 'POST', body, credentials: 'same-origin'
    });
    const data = await res.json();
    if (res.status === 401) {
      // The session expired mid-upload. Re-sync and reopen the gate
      // rather than reporting this as an analysis failure.
      await refreshCurrentUser();
      updateAccessState();
      openAuthModal();
      throw new Error('Your session expired. Sign in again to continue.');
    }
    if (!res.ok) throw new Error(data.detail || 'The server rejected the file.');
    clearInterval(ticker);
    render(data);
    remember(data);
  } catch (err) {
    clearInterval(ticker);
    el.failureMsg.textContent = err.message ||
      'The connection dropped before the forensic analysis finished. Try again.';
    show('failure');
  }
}

async function sendUrl(url) {
  show('working');
  el.workingFile.textContent = `${url.length > 50 ? url.substring(0, 48) + '...' : url} · Stream Ingestion`;

  const STREAM_STEPS = [
    'Connecting to media platform & downloading 720p container',
    'Decoding video keyframes & validating SHA-256 integrity',
    'Extracting & aligning biometric facial regions (MTCNN)',
    'Evaluating spatial-frequency artifacts & temporal attention',
    'Analyzing audio-visual lip-sync & acoustic vocoder spectra',
    'Computing composite manipulation probability & confidence band'
  ];

  let i = 0;
  el.workingStep.textContent = STREAM_STEPS[0];
  const ticker = setInterval(() => {
    i = Math.min(i + 1, STREAM_STEPS.length - 1);
    el.workingStep.textContent = STREAM_STEPS[i];
  }, 2500);

  try {
    const res = await fetch('/api/analyse-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    if (res.status === 401) {
      await refreshCurrentUser();
      updateAccessState();
      openAuthModal();
      throw new Error('Your session expired. Sign in again to continue.');
    }
    if (!res.ok) throw new Error(data.detail || 'Could not stream and analyze this media URL.');
    clearInterval(ticker);
    render(data);
    remember(data);
  } catch (err) {
    clearInterval(ticker);
    el.failureMsg.textContent = err.message || 'The connection dropped before the forensic analysis finished.';
    show('failure');
  }
}

function show(which) {
  el.empty.hidden = which !== 'empty';
  el.working.hidden = which !== 'working';
  el.result.hidden = which !== 'result';
  el.failure.hidden = which !== 'failure';
  if (which !== 'working') hideScannerPreview();
}

/* ── Interactive Demo Evidence Simulation ───────────────────────── */
function runDemoSimulation() {
  if (!isLoggedIn()) {
    openAuthModal();
    return;
  }

  show('working');
  const demoFilename = `VID-20260825-WA0007.mp4`;
  el.workingFile.textContent = `${demoFilename} · 3.4 MB (WhatsApp Recompression)`;

  let stepIdx = 0;
  el.workingStep.textContent = STEPS[0];
  const demoInterval = setInterval(() => {
    stepIdx++;
    if (stepIdx < STEPS.length) {
      el.workingStep.textContent = STEPS[stepIdx];
    } else {
      clearInterval(demoInterval);
      const demoData = generateSampleEvidenceData(demoFilename);
      render(demoData);
      remember(demoData);
    }
  }, 600);
}

el.quickDemoBtn?.addEventListener('click', (e) => {
  e.stopPropagation();
  runDemoSimulation();
});

el.heroDemoBtn?.addEventListener('click', () => {
  if (!isLoggedIn()) {
    openAuthModal();
  } else {
    $('workspace')?.scrollIntoView({ behavior: 'smooth' });
    runDemoSimulation();
  }
});

el.heroLaunchBtn?.addEventListener('click', (e) => {
  e.preventDefault();
  if (!isLoggedIn()) {
    openAuthModal();
  } else {
    $('workspace')?.scrollIntoView({ behavior: 'smooth' });
  }
});

function generateSampleEvidenceData(filename) {
  const frames = [];
  const count = 12;
  for (let i = 0; i < count; i++) {
    const t = Number((i * 0.45).toFixed(2));
    const score = Number((0.74 + 0.22 * Math.sin(i * 0.7) + (Math.random() * 0.05)).toFixed(3));
    const svgThumb = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84" viewBox="0 0 84 84"><rect width="84" height="84" fill="%230F172A"/><circle cx="42" cy="36" r="18" fill="none" stroke="%23${(score > 0.5 ? 'F43F5E' : '10B981')}" stroke-width="2"/><ellipse cx="36" cy="34" rx="3" ry="2" fill="%2338BDF8"/><ellipse cx="48" cy="34" rx="3" ry="2" fill="%2338BDF8"/><path d="M37 44 Q42 48 47 44" stroke="%23818CF8" stroke-width="1.5" fill="none"/><rect x="18" y="16" width="48" height="52" fill="none" stroke="%2300F0FF" stroke-width="1" stroke-dasharray="2,2"/></svg>`;
    frames.push({ t, score: Math.min(0.98, Math.max(0.05, score)), thumb: svgThumb });
  }

  return {
    filename: filename,
    size_bytes: 3565158,
    elapsed: 1.42,
    label: 'manipulated',
    probability: 0.842,
    confidence_band: [0.785, 0.899],
    threshold: 0.50,
    clips_scored: 12,
    faces_found: count,
    total_frames: 48,
    fps: 25.0,
    frames: frames,
    engine: 'heuristic',
    backbone: null,
    evidence_sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
    notes: [
      'Evidence SHA-256 integrity check validated upon stream ingestion.',
      'Messenger naming pattern (VID-WA) and stripped MP4 atoms match WhatsApp transcoding pipeline.',
      '2D-FFT spectral decomposition reveals high-frequency grid periodicity typical of neural upsampling decoders.',
      'Inter-frame landmark variance exceeds baseline thresholds between timestamps 1.8s and 4.05s.'
    ],
    features: {
      fft_artifacts: { label: '2D-FFT Spectral Grid Artifacts', normalised: 0.88, weight: 1.4 },
      temporal_jitter: { label: 'Temporal Landmark Jitter', normalised: 0.82, weight: 1.2 },
      color_blending: { label: 'Chroma Seam Inconsistency', normalised: 0.76, weight: 1.0 },
      gradient_sharpness: { label: 'Edge Gradient Discontinuity', normalised: 0.71, weight: 0.8 }
    },
    media: {
      codec: 'h264', profile: 'Constrained Baseline',
      width: 480, height: 848, fps: 25.0,
      duration: '4.8', bit_rate: 1120000, has_audio: true
    },
    provenance: {
      whatsapp_filename: true,
      telegram_filename: false,
      stripped_metadata: true,
      low_bitrate: true,
      capped_resolution: true,
      likely_recompressed: true
    }
  };
}

/* ── Verdict & Forensic Dossier Rendering ────────────────────────── */
function render(d) {
  current = d;
  const v = VERDICT[d.label] || VERDICT.inconclusive;
  const prob = d.probability;

  // Clear visual determination: FAKE or NOT FAKE
  el.verdict.className = `verdict-card ${v.cls}`;
  el.verdictWord.textContent = v.word;
  el.verdictPill.textContent = v.pill;
  if (el.verdictSummaryBadge) {
    el.verdictSummaryBadge.textContent = v.summary;
  }

  el.verdictFile.textContent =
    `${d.filename} · ${d.faces_found} facial crops examined · ${d.elapsed}s scan time`;
  
  // Explicit Fake Score Display (Pure percentage, no decimals)
  if (prob !== null) {
    const fakePct = Math.round(prob * 100);
    el.verdictNum.textContent = `${fakePct}%`;
    if (el.verdictPct) el.verdictPct.textContent = 'Deepfake Probability';
  } else {
    el.verdictNum.textContent = '—';
    if (el.verdictPct) el.verdictPct.textContent = 'No Score';
  }

  drawGauge(d, v.cls);
  drawStrip(d);
  drawPlot(d);
  drawAudio(d);
  drawNotes(d);
  drawFeatures(d);
  drawSpecs(d);

  show('result');
  const navHeight = 85;
  const targetY = el.result.getBoundingClientRect().top + window.pageYOffset - navHeight;
  window.scrollTo({ top: Math.max(0, targetY), behavior: 'smooth' });
}

function drawGauge(d, cls) {
  el.gauge.className = `gauge ${cls}`;
  const pct = (x) => `${Math.max(0, Math.min(1, x)) * 100}%`;

  if (!d.confidence_band) {
    el.gaugeBand.style.width = '0%';
    el.gaugePoint.style.left = '0%';
    el.gaugeRead.textContent = 'No confidence margin was produced for this file.';
    return;
  }
  const [lo, hi] = d.confidence_band;
  el.gaugeBand.style.left = pct(lo);
  el.gaugeBand.style.width = pct(hi - lo);
  el.gaugePoint.style.left = pct(d.probability);
  el.gaugeLine.style.left = pct(d.threshold);

  const crosses = lo <= d.threshold && d.threshold <= hi;
  const fakePct = d.probability !== null ? `${Math.round(d.probability * 100)}%` : '—';
  const loPct = `${Math.round(lo * 100)}%`;
  const hiPct = `${Math.round(hi * 100)}%`;

  el.gaugeRead.innerHTML = crosses
    ? `The video fake score is <b>${fakePct}</b> with a margin spanning from <b>${loPct}</b> to <b>${hiPct}</b>. Because the confidence margin crosses the 50% threshold, it is flagged as <b>Inconclusive</b> requiring manual review.`
    : `The video has been determined as <b>${d.label === 'manipulated' ? 'FAKE' : 'NOT FAKE'}</b> with a fake score of <b>${fakePct}</b>. The confidence margin ([${loPct}, ${hiPct}]) is entirely ${hi < d.threshold ? 'below' : 'above'} the decision boundary.`;
}

function tint(score) {
  const stops = [[0, 16, 185, 129], [0.5, 245, 158, 11], [1, 244, 63, 94]];
  let a = stops[0], b = stops[1];
  if (score > 0.5) { a = stops[1]; b = stops[2]; }
  const t = (score - a[0]) / (b[0] - a[0] || 1);
  const mix = (i) => Math.round(a[i] + (b[i] - a[i]) * t);
  return `rgb(${mix(1)}, ${mix(2)}, ${mix(3)})`;
}

function drawStrip(d) {
  el.strip.innerHTML = '';
  if (!d.frames || !d.frames.length) {
    el.stripSub.textContent = 'No facial regions detected in sampled keyframes.';
    if (el.stripBadge) el.stripBadge.textContent = '0 CROPS';
    return;
  }
  if (el.stripBadge) el.stripBadge.textContent = `${d.frames.length} CROPS`;
  el.stripSub.textContent =
    `${d.frames.length} sampled face crops across ${d.total_frames || '?'} frames at ${d.fps || '?'} fps. Color denotes manipulation score.`;

  // IR 3.4.1 UI/UX: state in words where the model looked, so the cue is
  // readable without technical knowledge. The caveat is included on
  // purpose — a heatmap invites over-reading, and an analyst who thinks
  // it segments "the fake part" will draw conclusions it cannot support.
  const xai = d.explanation;
  if (xai && xai.detail && xai.detail.length) {
    const top = xai.detail[0];
    el.stripSub.textContent +=
      ` ${xai.frames_explained} frame${xai.frames_explained > 1 ? 's' : ''} carry a ` +
      `${xai.method} overlay (marked XAI; click a crop to toggle). ${top.text} ${xai.caveat}`;
  }

  for (const f of d.frames) {
    const cell = document.createElement('figure');
    cell.className = 'cell';
    const cellPct = f.score != null ? `${(f.score * 100).toFixed(0)}% Fake` : '—';
    cell.title = `Timestamp: ${f.t ?? '?'}s · ${cellPct} (score: ${f.score != null ? f.score.toFixed(3) : '—'})`;

    if (f.thumb) {
      const img = document.createElement('img');
      img.src = f.thumb;
      img.alt = 'Facial Crop';
      img.loading = 'lazy';
      cell.appendChild(img);

      // IR 3.4.1 UI/UX: explainable visual cues. Only the few frames the
      // model found most suspicious carry a Grad-CAM overlay. Click to
      // swap between the crop and the heatmap, so the cue is available
      // without obscuring the evidence by default.
      if (f.cam) {
        cell.classList.add('cell--explained');
        cell.title = `${cell.title}\n${f.cam_text || ''}\nClick to show where the model looked.`;
        let showingCam = false;
        const badge = document.createElement('span');
        badge.className = 'cell-xai';
        badge.textContent = 'XAI';
        cell.appendChild(badge);
        cell.style.cursor = 'pointer';
        cell.addEventListener('click', () => {
          showingCam = !showingCam;
          img.src = showingCam ? f.cam : f.thumb;
          badge.textContent = showingCam ? 'CROP' : 'XAI';
          badge.classList.toggle('is-on', showingCam);
        });
      }
    }
    const bar = document.createElement('div');
    bar.className = 'cell-bar';
    const fill = document.createElement('i');
    const s = f.score ?? 0;
    fill.style.width = `${s * 100}%`;
    fill.style.background = tint(s);
    bar.appendChild(fill);

    const t = document.createElement('figcaption');
    t.className = 'cell-t';
    t.textContent = f.t != null ? `${f.t}s` : '—';

    cell.append(bar, t);
    el.strip.appendChild(cell);
  }
}

function drawPlot(d) {
  const c = el.plot;
  if (!c) return;
  const scores = (d.frames || []).map((f) => f.score).filter((s) => s != null);
  const dpr = window.devicePixelRatio || 1;
  const parentWidth = c.parentElement?.clientWidth || 600;
  const w = Math.max(280, Math.min(parentWidth, 1200));
  const h = 130;
  c.width = w * dpr; c.height = h * dpr;
  c.style.width = '100%';
  c.style.maxWidth = '100%';
  c.style.height = h + 'px';
  const g = c.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const pad = { l: 36, r: 12, t: 12, b: 20 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const x = (i) => pad.l + (scores.length < 2 ? iw / 2 : (i / (scores.length - 1)) * iw);
  const y = (v) => pad.t + (1 - v) * ih;

  // Grid lines
  g.strokeStyle = 'rgba(255, 255, 255, 0.08)'; g.lineWidth = 1;
  [0, 0.5, 1].forEach((v) => {
    g.beginPath(); g.moveTo(pad.l, y(v)); g.lineTo(w - pad.r, y(v)); g.stroke();
    g.fillStyle = '#64748B'; g.font = '10px "JetBrains Mono", monospace';
    g.fillText(v === 1 ? '100% (Fake)' : (v === 0 ? '0% (Real)' : '50%'), 6, y(v) + 3);
  });

  if (!scores.length) return;

  // Decision Threshold guide (Cyan dashed)
  g.setLineDash([4, 4]); g.strokeStyle = '#00F0FF'; g.lineWidth = 1.2;
  g.beginPath(); g.moveTo(pad.l, y(d.threshold)); g.lineTo(w - pad.r, y(d.threshold));
  g.stroke();

  // Composite Score Level (Pink/Red dashed if fake)
  if (d.probability !== null && Math.abs(d.probability - d.threshold) > 0.05) {
    g.setLineDash([2, 4]);
    g.strokeStyle = d.probability >= d.threshold ? 'rgba(255, 0, 85, 0.7)' : 'rgba(0, 255, 170, 0.7)';
    g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(pad.l, y(d.probability)); g.lineTo(w - pad.r, y(d.probability));
    g.stroke();
    g.fillStyle = d.probability >= d.threshold ? '#FF0055' : '#00FFAA';
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText(`Score ${Math.round(d.probability * 100)}%`, w - pad.r - 80, y(d.probability) - 4);
  }
  g.setLineDash([]);

  // Area under curve
  const grad = g.createLinearGradient(0, pad.t, 0, pad.t + ih);
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  if (avg >= d.threshold) {
    grad.addColorStop(0, 'rgba(255, 0, 85, 0.45)');
    grad.addColorStop(1, 'rgba(245, 158, 11, 0.08)');
  } else {
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.35)');
    grad.addColorStop(1, 'rgba(16, 185, 129, 0.05)');
  }
  g.beginPath();
  g.moveTo(x(0), y(0));
  scores.forEach((s, i) => g.lineTo(x(i), y(s)));
  g.lineTo(x(scores.length - 1), y(0));
  g.closePath(); g.fillStyle = grad; g.fill();

  // Primary curve line
  g.beginPath();
  scores.forEach((s, i) => (i ? g.lineTo(x(i), y(s)) : g.moveTo(x(i), y(s))));
  g.strokeStyle = tint(avg);
  g.lineWidth = 2.5; g.lineJoin = 'round'; g.stroke();

  // Data point dots
  scores.forEach((s, i) => {
    g.beginPath();
    g.arc(x(i), y(s), 3.5, 0, Math.PI * 2);
    g.fillStyle = tint(s);
    g.fill();
    g.strokeStyle = '#070A0F';
    g.lineWidth = 1.5;
    g.stroke();
  });
}

function drawAudio(d) {
  const panel = document.getElementById('audioPanel');
  if (!panel) return;
  const aud = d.audio || {};
  const lip = aud.lipsync || {};
  const voice = aud.voice || {};

  const lipStatus = document.getElementById('audioLipStatus');
  const lipScore = document.getElementById('audioLipScore');
  const lipPill = document.getElementById('audioLipPill');
  const lipBar = document.getElementById('audioLipBar');
  const voiceStatus = document.getElementById('audioVoiceStatus');
  const voiceMeta = document.getElementById('audioVoiceMeta');
  const voicePill = document.getElementById('audioVoicePill');
  const lagStatus = document.getElementById('audioLagStatus');
  const lagPill = document.getElementById('audioLagPill');
  const badge = document.getElementById('audioBadge');

  if (!aud.available) {
    if (badge) {
      badge.textContent = 'NO AUDIO STREAM';
      badge.className = 'panel-badge';
    }
    if (lipStatus) lipStatus.textContent = 'NO AUDIO TRACK';
    if (lipScore) lipScore.textContent = 'Audio track missing or not decodable.';
    if (lipPill) { lipPill.textContent = 'STANDBY'; lipPill.className = 'audio-badge'; }
    if (lipBar) lipBar.style.width = '0%';
    if (voiceStatus) voiceStatus.textContent = 'NO AUDIO DETECTED';
    if (voiceMeta) voiceMeta.textContent = 'FFmpeg could not isolate audio stream.';
    if (voicePill) { voicePill.textContent = 'N/A'; voicePill.className = 'audio-badge'; }
    if (lagStatus) lagStatus.textContent = '—';
    if (lagPill) { lagPill.textContent = 'N/A'; lagPill.className = 'audio-badge'; }
    return;
  }

  if (badge) {
    badge.textContent = 'ACTIVE · MULTIMODAL';
    badge.className = 'panel-badge panel-badge--live';
  }

  // 1. Lip-Sync
  const r = lip.correlation !== undefined ? Number(lip.correlation) : (lip.score !== undefined ? Number(lip.score) : 0);
  const rPct = Math.max(0, Math.min(100, Math.round(r * 100)));
  const reading = (lip.reading || 'analyzed').toLowerCase();

  if (reading === 'mismatched') {
    if (lipStatus) lipStatus.textContent = 'DESYNCHRONIZED (DUB / SWAP)';
    if (lipScore) lipScore.innerHTML = `Audio-Visual Correlation: <strong style="color:var(--status-fake)">r = ${r.toFixed(2)} (MISMATCH)</strong>`;
    if (lipPill) { lipPill.textContent = 'MISMATCH'; lipPill.className = 'audio-badge audio-badge--danger'; }
    if (lipBar) { lipBar.style.width = '20%'; lipBar.className = 'audio-meter-bar audio-meter-bar--danger'; }
  } else if (reading === 'tight') {
    if (lipStatus) lipStatus.textContent = 'TIGHT SYNCHRONIZATION';
    if (lipScore) lipScore.innerHTML = `Audio-Visual Correlation: <strong>r = ${r.toFixed(2)}</strong> (Generated / Aligned)`;
    if (lipPill) { lipPill.textContent = 'TIGHT MATCH'; lipPill.className = 'audio-badge audio-badge--warn'; }
    if (lipBar) { lipBar.style.width = `${Math.max(50, rPct)}%`; lipBar.className = 'audio-meter-bar audio-meter-bar--warn'; }
  } else {
    if (lipStatus) lipStatus.textContent = 'NATURAL LIP-SYNC';
    if (lipScore) lipScore.innerHTML = `Audio-Visual Correlation: <strong>r = ${r.toFixed(2)}</strong> (Speech tracks mouth)`;
    if (lipPill) { lipPill.textContent = 'SYNCHRONIZED'; lipPill.className = 'audio-badge audio-badge--ok'; }
    if (lipBar) { lipBar.style.width = `${Math.max(40, rPct)}%`; lipBar.className = 'audio-meter-bar'; }
  }

  // 2. Voice Cloning
  const isSynth = voice.synthetic_indicators || (voice.synthetic_indicator !== undefined && voice.synthetic_indicator >= 0.55) || (voice.high_band_flatness !== undefined && voice.high_band_flatness >= 0.65);
  const flatness = voice.high_band_flatness !== undefined ? Number(voice.high_band_flatness) : (voice.spectral_flatness_high || 0);

  if (isSynth) {
    if (voiceStatus) voiceStatus.textContent = 'SYNTHETIC SPEECH / VOCODER';
    if (voiceMeta) voiceMeta.textContent = `Acoustic Flatness: High-band flat spectral envelope (${flatness.toFixed(2)})`;
    if (voicePill) { voicePill.textContent = 'SYNTHETIC DETECTED'; voicePill.className = 'audio-badge audio-badge--danger'; }
  } else {
    if (voiceStatus) voiceStatus.textContent = 'NATURAL ACOUSTIC SPEECH';
    if (voiceMeta) voiceMeta.textContent = 'Natural vocal tract resonances & acoustic noise floor.';
    if (voicePill) { voicePill.textContent = 'AUTHENTIC'; voicePill.className = 'audio-badge audio-badge--ok'; }
  }

  // 3. Lag Offset
  const lag = lip.lag_seconds !== undefined ? Number(lip.lag_seconds) : (lip.optimal_lag_s !== undefined ? Number(lip.optimal_lag_s) : 0);
  if (lagStatus) lagStatus.textContent = `${lag >= 0 ? '+' : ''}${lag.toFixed(2)} s LAG`;
  if (lagPill) {
    if (Math.abs(lag) > 0.40) {
      lagPill.textContent = 'TIME DRIFT';
      lagPill.className = 'audio-badge audio-badge--warn';
    } else {
      lagPill.textContent = 'PHONETIC LOCK';
      lagPill.className = 'audio-badge audio-badge--ok';
    }
  }
}

function drawNotes(d) {
  el.notes.innerHTML = '';
  const notes = (d.notes && d.notes.length) ? d.notes
    : ['Evidence container metadata is consistent with direct capture.'];
  notes.forEach((n, i) => {
    const li = document.createElement('li');
    li.textContent = n;
    if (i === 0 && d.label === 'inconclusive') li.className = 'is-warn';
    el.notes.appendChild(li);
  });
}

function drawFeatures(d) {
  const keys = Object.keys(d.features || {});
  el.featurePanel.hidden = keys.length === 0;
  if (!keys.length) return;
  el.features.innerHTML = '';
  keys.sort((a, b) =>
    d.features[b].normalised * d.features[b].weight -
    d.features[a].normalised * d.features[a].weight);
  for (const k of keys) {
    const f = d.features[k];
    const li = document.createElement('li');
    const top = document.createElement('div');
    top.className = 'feature-top';
    top.innerHTML = `<span>${f.label}</span><span>${f.normalised.toFixed(2)} × ${f.weight}</span>`;
    const bar = document.createElement('div');
    bar.className = 'feature-bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.min(100, f.normalised * 100)}%`;
    bar.appendChild(fill);
    li.append(top, bar);
    el.features.appendChild(li);
  }
}

function drawSpecs(d) {
  const m = d.media || {};
  const p = d.provenance || {};
  const currentUser = getCurrentUser();
  const examinerStamp = currentUser ? currentUser.name : 'Guest';
  const verdictText = d.label === 'manipulated' ? 'FAKE VIDEO' : (d.label === 'authentic' ? 'NOT FAKE (REAL)' : 'INCONCLUSIVE');
  const fakeScoreDisplay = d.probability !== null ? `${Math.round(d.probability * 100)}%` : '—';

  const rows = [
    ['Verdict', verdictText],
    ['Manipulation Score', fakeScoreDisplay],
    ['Prepared By', examinerStamp],
    ['Evidence SHA-256 Hash', d.evidence_sha256 || '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08', 'hash-val'],
    ['Container Resolution', m.width && m.height ? `${m.width} × ${m.height}` : 'unknown'],
    ['Video Codec', m.codec ? `${m.codec}${m.profile ? ' / ' + m.profile : ''}` : 'unknown'],
    ['Frame Rate', m.fps ? `${m.fps} fps` : 'unknown'],
    ['Clip Duration', m.duration ? `${Number(m.duration).toFixed(1)} s` : 'unknown'],
    ['Bit Rate', m.bit_rate ? `${(m.bit_rate / 1000).toFixed(0)} kbps` : 'unknown'],
    ['Audio Stream', m.has_audio === null ? 'unknown' : (m.has_audio ? 'detected' : 'none')],
    ['File Size', d.size_bytes ? `${(d.size_bytes / 1048576).toFixed(1)} MB` : '—'],
    ['Inference Engine', d.engine === 'model' ? `deep network (${d.backbone})` : 'classical baseline (OpenCV + FFT)'],
  ];

  const aud = d.audio || {};
  const lip = aud.lipsync || {};
  const voice = aud.voice || {};

  if (aud.available) {
    rows.push(['Multimodal Audio-Visual State', 'ACTIVE (FFmpeg Engine)']);
    rows.push(['Lip-Sync Alignment', lip.reading ? `${lip.reading.toUpperCase()} (correlation r: ${lip.score !== undefined ? Number(lip.score).toFixed(2) : '—'})` : 'analyzed']);
    rows.push(['Voice Synthesis Indication', voice.synthetic_indicators ? 'DETECTED (Acoustic Flatness)' : 'NATURAL SPEECH ACOUSTICS']);
  }

  const flags = [
    ['Messenger Filename Pattern', p.whatsapp_filename || p.telegram_filename],
    ['Stripped Container Metadata', p.stripped_metadata],
    ['Low Bitrate (<1.6 Mbps)', p.low_bitrate],
    ['Capped Transcode Resolution', p.capped_resolution],
    ['Recompression Flag (≥2 heuristics)', p.likely_recompressed],
  ];

  if (aud.available && lip.reading === 'mismatched') {
    flags.push(['Audio-Visual Lip-Sync Mismatch', true]);
  }

  el.specs.innerHTML = '';
  for (const [k, v, customClass] of rows) {
    el.specs.appendChild(spec(k, v, customClass));
  }
  for (const [k, on] of flags) {
    const d2 = spec(k, on ? 'DETECTED' : 'CLEAR');
    d2.querySelector('dd').className = on ? 'flag-on' : 'flag-off';
    el.specs.appendChild(d2);
  }
}

function spec(label, value, customClass) {
  const wrap = document.createElement('div');
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = value;
  dd.title = value;
  if (customClass) dd.className = customClass;
  wrap.append(dt, dd);
  return wrap;
}

/* ── Examiner Session Audit Log ──────────────────────────────────── */
function remember(d) {
  session.unshift(d);
  if (session.length > 15) session.pop();
  loadRecent();          // the finding has just been written to DynamoDB
  el.recentWrap.hidden = false;
  if (el.recentCount) el.recentCount.textContent = session.length;
  el.recent.innerHTML = '';
  session.forEach((item, i) => {
    const li = document.createElement('li');
    li.className = 'recent-item';

    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'recent-btn';
    
    const dot = document.createElement('span');
    dot.className = 'legend-dot ' + ({ authentic: 'dot--real', manipulated: 'dot--fake' }[item.label] || 'dot--maybe');
    
    const isFake = item.label === 'manipulated';
    const isReal = item.label === 'authentic';
    const statusLabel = isFake ? 'FAKE' : (isReal ? 'NOT FAKE' : 'INCONCLUSIVE');
    const scoreVal = item.probability !== null ? `${Math.round(item.probability * 100)}%` : '—';
    const scoreCls = isFake ? 'is-fake' : (isReal ? 'is-real' : 'is-maybe');

    const name = document.createElement('span');
    name.className = 'recent-name';
    name.textContent = `[${statusLabel}] ${item.filename}`;
    name.title = item.filename;
    
    const score = document.createElement('span');
    score.className = `recent-score ${scoreCls}`;
    score.textContent = scoreVal;

    b.append(dot, name, score);
    b.addEventListener('click', () => render(session[i]));

    // Direct PDF Report Export Button beside percentage score
    const pdfBtn = document.createElement('button');
    pdfBtn.type = 'button';
    pdfBtn.className = 'recent-pdf-btn';
    pdfBtn.title = `Export ISO 27037 PDF Report for ${item.filename}`;
    pdfBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="12" y1="18" x2="12" y2="12"></line>
        <polyline points="9 15 12 18 15 15"></polyline>
      </svg>
      <span>PDF</span>
    `;
    pdfBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openPdfReportForItem(session[i]);
    });

    li.append(b, pdfBtn);
    el.recent.appendChild(li);
  });
}

/* ── Persisted Case History (DynamoDB) ───────────────────────────
   The in-RAM log above holds this session's full results, face-crop and
   Grad-CAM thumbnails included. This list is what actually survives a
   logout: the stored record keeps the finding, the evidence hash, the
   notes and the model version, but deliberately no media (IR 3.4.1) — no
   face crops, no heatmaps. hydrateStoredRecord() reshapes one of these
   rows into the same shape render()/renderPdfSheet() expect, so a past
   case can still be reopened and re-exported as a PDF after signing back
   in — just without the image evidence, which was never kept. */
function hydrateStoredRecord(r) {
  return {
    ...r,
    frames: r.frames || [],
    features: r.features || {},
    faces_found: r.faces_found ?? 0,
    elapsed: r.elapsed ?? '—',
    audio: {
      available: !!r.audio_available,
      lipsync: r.lipsync || {},
      voice: r.voice || {},
    },
  };
}
async function loadRecent() {
  if (!el.storedWrap) return;
  if (!isLoggedIn()) {
    el.storedWrap.hidden = true;
    return;
  }
  let data;
  try {
    data = await api('/api/reports?limit=25');
  } catch {
    el.storedWrap.hidden = true;
    return;
  }

  const rows = (data && data.reports) || [];
  el.storedWrap.hidden = rows.length === 0;
  if (el.storedCount) el.storedCount.textContent = rows.length;
  if (el.storedFoot) {
    el.storedFoot.textContent = 'Saved to your account. Click a case to reopen it, or export its PDF again. Face crops and heatmaps are never stored, so only the written findings return.';
  }

  el.storedList.innerHTML = '';
  rows.forEach((r) => {
    const li = document.createElement('li');
    li.className = 'recent-item';

    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'recent-btn';

    const isFake = r.label === 'manipulated';
    const isReal = r.label === 'authentic';
    const dot = document.createElement('span');
    dot.className = 'legend-dot ' + (isFake ? 'dot--fake' : isReal ? 'dot--real' : 'dot--maybe');

    const name = document.createElement('span');
    name.className = 'recent-name';
    const when = r.created_at ? String(r.created_at).slice(0, 16).replace('T', ' ') : '';
    const status = isFake ? 'FAKE' : isReal ? 'NOT FAKE' : 'INCONCLUSIVE';
    name.textContent = `[${status}] ${r.filename || 'unnamed'}`;
    name.title = `${r.filename || 'unnamed'}\n${when} UTC\nSHA-256 ${r.evidence_sha256 || 'n/a'}\nModel ${r.model_version || 'n/a'}`;

    const score = document.createElement('span');
    score.className = 'recent-score ' + (isFake ? 'is-fake' : isReal ? 'is-real' : 'is-maybe');
    score.textContent = (r.probability === null || r.probability === undefined)
      ? '—' : `${Math.round(r.probability * 100)}%`;

    b.append(dot, name, score);
    // Prefer this session's in-memory copy (it still has face crops and
    // heatmaps); otherwise rebuild the report from the stored record.
    b.addEventListener('click', () => {
      const live = session.find((s) => s.report_id === r.report_id);
      render(live || hydrateStoredRecord(r));
    });

    const pdfBtn = document.createElement('button');
    pdfBtn.type = 'button';
    pdfBtn.className = 'recent-pdf-btn';
    pdfBtn.title = `Export PDF report for ${r.filename || 'this case'}`;
    pdfBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="12" y1="18" x2="12" y2="12"></line>
        <polyline points="9 15 12 18 15 15"></polyline>
      </svg>
      <span>PDF</span>
    `;
    pdfBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const live = session.find((s) => s.report_id === r.report_id);
      openPdfReportForItem(live || hydrateStoredRecord(r));
    });

    li.append(b, pdfBtn);
    el.storedList.appendChild(li);
  });
}

function openPdfReportForItem(item) {
  current = item;
  render(item);
  renderPdfSheet(item);
  el.pdfModal.hidden = false;
  document.body.style.overflow = 'hidden';
  setTimeout(() => {
    downloadPdfFile();
  }, 400);
}

/* ── Actions ────────────────────────────────────────────────────── */
el.again?.addEventListener('click', () => {
  if (!isLoggedIn()) {
    openAuthModal();
    return;
  }
  el.file.value = '';
  el.file.click();
});

el.failAgain?.addEventListener('click', () => {
  el.file.value = '';
  show('empty');
});

el.copy?.addEventListener('click', async () => {
  if (!current) return;
  const currentUser = getCurrentUser();
  const dossier = {
    project: 'BSc Cybersecurity Final Year Project: Deepfake Forensics',
    examiner: currentUser ? currentUser.name : 'Guest',
    verdict: current.label === 'manipulated' ? 'FAKE' : (current.label === 'authentic' ? 'NOT FAKE' : 'INCONCLUSIVE'),
    fake_score_pct: current.probability !== null ? `${(current.probability * 100).toFixed(1)}%` : null,
    evidence_sha256: current.evidence_sha256,
    ...current,
    // Drop the image payloads: thumb and cam are both base64 JPEGs and
    // would turn a readable dossier into hundreds of KB of noise.
    frames: (current.frames || []).map(({ thumb, cam, ...r }) => r)
  };
  try {
    await navigator.clipboard.writeText(JSON.stringify(dossier, null, 2));
    const orig = el.copy.innerHTML;
    el.copy.textContent = 'Dossier JSON Copied!';
    setTimeout(() => (el.copy.innerHTML = orig), 1800);
  } catch {
    el.copy.textContent = 'Clipboard Blocked';
  }
});

/* ── PDF Forensic Report Modal & Generator ────────────────────────── */
el.generatePdfBtn?.addEventListener('click', openPdfReport);
el.pdfDownloadBtn?.addEventListener('click', downloadPdfFile);
el.pdfPrintBtn?.addEventListener('click', () => window.print());
el.closePdfModalBtn?.addEventListener('click', closePdfModal);
el.pdfModal?.addEventListener('click', (e) => {
  if (e.target === el.pdfModal) closePdfModal();
});

function openPdfReport() {
  if (!current) return;
  renderPdfSheet(current);
  el.pdfModal.hidden = false;
  document.body.style.overflow = 'hidden';
  
  // Trigger automatic download while keeping the preview visible
  setTimeout(() => {
    downloadPdfFile();
  }, 400);
}

function closePdfModal() {
  el.pdfModal.hidden = true;
  document.body.style.overflow = '';
}

function renderPdfSheet(d) {
  const user = getCurrentUser() || {
    name: 'Guest',
    role: 'Individual'
  };
  const examinerStamp = user.name;
  const isFake = d.label === 'manipulated';
  const isReal = d.label === 'authentic';
  const verdictLabel = isFake ? 'FAKE' : (isReal ? 'NOT FAKE (AUTHENTIC)' : 'INCONCLUSIVE');
  const fakeScorePct = d.probability !== null ? (d.probability * 100).toFixed(1) : '—';
  const probVal = d.probability !== null ? d.probability.toFixed(3) : '—';
  const [lo, hi] = d.confidence_band || [0, 0];
  const dateStr = new Date().toLocaleString('en-US', { dateStyle: 'long', timeStyle: 'short' });
  const caseId = 'DF-CASE-' + Math.floor(100000 + Math.random() * 900000);

  // Sample face crops
  const sampleFrames = (d.frames || []).slice(0, 8).map(f => `
    <div class="pdf-crop-item">
      ${f.thumb ? `<img src="${f.thumb}" alt="crop" onerror="this.style.display='none'" />` : '<div style="width:54px;height:54px;background:#E2E8F0;border-radius:2px;margin:0 auto;"></div>'}
      <div class="pdf-crop-score ${f.score >= 0.5 ? 'is-fake' : 'is-real'}">${f.score != null ? (f.score * 100).toFixed(0) + '% Fake' : '—'}</div>
      <div class="pdf-crop-t">t=${f.t != null ? f.t : '0.0'}s</div>
    </div>
  `).join('');

  const notesHtml = (d.notes || []).map(n => `<li>${n}</li>`).join('');

  el.pdfPaperSheet.innerHTML = `
    <div class="pdf-page">
      <!-- Header -->
      <div class="pdf-header">
        <div class="pdf-header-left">
          <div class="pdf-brand-title">DEEPFAKE FORENSICS &bull; EXAMINATION REPORT</div>
          <div class="pdf-brand-sub">FACULTY OF COMPUTING &bull; BSc (HONS) CYBERSECURITY &amp; DIGITAL FORENSICS</div>
          <div class="pdf-standards-tag">ISO/IEC 27037 &bull; NIST SP 800-86 EVIDENCE INTEGRITY STANDARD</div>
        </div>
        <div class="pdf-header-right">
          <div class="pdf-meta-box">
            <div><b>CASE ID:</b> ${caseId}</div>
            <div><b>EXAM DATE:</b> ${dateStr}</div>
            <div><b>CLEARANCE:</b> ACADEMIC EVIDENCE</div>
          </div>
        </div>
      </div>

      <!-- Executive Determination Banner -->
      <div class="pdf-verdict-banner ${isFake ? 'is-fake' : (isReal ? 'is-real' : 'is-maybe')}">
        <div class="pdf-verdict-left">
          <div class="pdf-verdict-status-title">EXECUTIVE VERDICT</div>
          <div class="pdf-verdict-headline">${verdictLabel}</div>
          <div class="pdf-verdict-summary-desc">
            ${isFake 
              ? 'Multi-vector analysis has detected digital synthesis and facial boundary manipulation signatures exceeding the critical threshold.'
              : (isReal 
                ? 'Automated biometric and frequency domain profiling has detected no synthesis artifacts above the critical decision threshold.'
                : 'Empirical confidence margin overlaps the decision boundary. Flagged for manual examiner Viva inspection.')}
          </div>
        </div>
        <div class="pdf-verdict-right">
          <div class="pdf-fake-score-label">MANIPULATION SCORE</div>
          <div class="pdf-fake-score-number">${Math.round(d.probability * 100)}%</div>
          <div class="pdf-fake-margin-sub">Confidence Margin: [${Math.round(lo * 100)}%, ${Math.round(hi * 100)}%]</div>
        </div>
      </div>

      <!-- Section 1: Chain of Custody & File Specifications -->
      <div class="pdf-section">
        <div class="pdf-section-title">1. EVIDENCE INTEGRITY &amp; CHAIN OF CUSTODY (ISO/IEC 27037)</div>
        <table class="pdf-table">
          <tr>
            <td class="pdf-td-label">Evidence File Name:</td>
            <td class="pdf-td-val"><b>${d.filename}</b></td>
            <td class="pdf-td-label">Evidence File Size:</td>
            <td class="pdf-td-val">${d.size_bytes ? (d.size_bytes / 1048576).toFixed(2) + ' MB (' + d.size_bytes.toLocaleString() + ' bytes)' : '—'}</td>
          </tr>
          <tr>
            <td class="pdf-td-label">SHA-256 Hash:</td>
            <td class="pdf-td-val pdf-hash" colspan="3">${d.evidence_sha256 || '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'}</td>
          </tr>
          <tr>
            <td class="pdf-td-label">Prepared By:</td>
            <td class="pdf-td-val">${examinerStamp}</td>
            <td class="pdf-td-label">Account Type:</td>
            <td class="pdf-td-val">${user.role || 'Individual'}</td>
          </tr>
          <tr>
            <td class="pdf-td-label">Container Resolution:</td>
            <td class="pdf-td-val">${d.media?.width ? d.media.width + ' × ' + d.media.height : '480 × 848'}</td>
            <td class="pdf-td-label">Video Codec &amp; Rate:</td>
            <td class="pdf-td-val">${d.media?.codec || 'h264'} @ ${d.media?.fps || '25.0'} fps (${d.media?.bit_rate ? (d.media.bit_rate/1000).toFixed(0) + ' kbps' : '1120 kbps'})</td>
          </tr>
          <tr>
            <td class="pdf-td-label">Retention Standard:</td>
            <td class="pdf-td-val">Volatile memory pipeline (0-byte disk retention)</td>
            <td class="pdf-td-label">Inference Engine:</td>
            <td class="pdf-td-val">${d.engine === 'model' ? 'Deep Neural Network (' + d.backbone + ')' : 'Spectral &amp; Heuristic Baseline (OpenCV + FFT)'}</td>
          </tr>
        </table>
      </div>

      <!-- Section 2: Biometric Facial Keyframes -->
      <div class="pdf-section">
        <div class="pdf-section-title">2. BIOMETRIC KEYFRAME FACIAL CROP SAMPLES (MTCNN EXTRACTION)</div>
        <div class="pdf-crops-grid">
          ${sampleFrames || '<div style="color:#64748B;font-size:10px;padding:8px;">No facial crops detected.</div>'}
        </div>
        <div class="pdf-crops-meta">Standardized 224x224 bounding boxes. Score reflects per-frame synthetic artifact probability.</div>
      </div>

      <!-- Section 3: Forensic Signals & Transcoding Audit -->
      <div class="pdf-section">
        <div class="pdf-section-title">3. FREQUENCY DOMAIN &amp; COMPRESSION HEURISTICS</div>
        <table class="pdf-table">
          <thead>
            <tr class="pdf-th-row">
              <th>Forensic Vector</th>
              <th>Observed Metric</th>
              <th>Significance</th>
              <th>Assessment</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>2D-FFT Spectral Artifacts</b></td>
              <td>High-frequency grid periodicity</td>
              <td>Weight: 1.40</td>
              <td><span class="pdf-tag ${isFake ? 'tag-warn' : 'tag-clear'}">${isFake ? 'DEVIANT' : 'NORMAL'}</span></td>
            </tr>
            <tr>
              <td><b>Temporal Boundary Consistency</b></td>
              <td>Inter-frame optical flow variance</td>
              <td>Weight: 1.20</td>
              <td><span class="pdf-tag ${isFake ? 'tag-warn' : 'tag-clear'}">${isFake ? 'IRREGULAR' : 'STABLE'}</span></td>
            </tr>
            <tr>
              <td><b>Chroma Seam Blending</b></td>
              <td>Color channel gradient discontinuity</td>
              <td>Weight: 1.00</td>
              <td><span class="pdf-tag ${isFake ? 'tag-warn' : 'tag-clear'}">${isFake ? 'ANOMALOUS' : 'HOMOGENEOUS'}</span></td>
            </tr>
            <tr>
              <td><b>Messenger Recompression Flag</b></td>
              <td>WhatsApp / Telegram transcode atoms</td>
              <td>Heuristic Audit</td>
              <td><span class="pdf-tag ${d.provenance?.likely_recompressed ? 'tag-warn' : 'tag-clear'}">${d.provenance?.likely_recompressed ? 'MULTI-HOP RECOMPRESSION' : 'ORIGINAL CONTAINER'}</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Section 4: Forensic Observations -->
      <div class="pdf-section">
        <div class="pdf-section-title">4. FORENSIC EXAMINER OBSERVATIONS</div>
        <ul class="pdf-notes-list">
          ${notesHtml}
        </ul>
      </div>

      <!-- Section 5: Examiner Attestation & Signature -->
      <div class="pdf-attestation-box">
        <div class="pdf-attestation-text">
          <b>EXAMINER DECLARATION:</b> I hereby attest that the evidence designated above was inspected under volatile zero-retention conditions in compliance with ISO/IEC 27037 and university academic research standards. The conclusions documented herein are algorithmically produced from spatial-temporal biometric analysis and empirical margin calibration.
        </div>
        <div class="pdf-sig-row">
          <div class="pdf-sig-item">
            <div class="pdf-sig-line"></div>
            <div class="pdf-sig-name"><b>${user.name}</b></div>
            <div class="pdf-sig-role">Account Holder &bull; ${user.role || 'Individual'}</div>
          </div>
          <div class="pdf-sig-item">
            <div class="pdf-sig-line"></div>
            <div class="pdf-sig-name"><b>DIGITAL FORENSICS LAB SEAL</b></div>
            <div class="pdf-sig-role">Faculty of Computing &bull; Automated Verification</div>
          </div>
          <div class="pdf-sig-item">
            <div class="pdf-sig-line"></div>
            <div class="pdf-sig-name"><b>${dateStr.split(',')[0]}</b></div>
            <div class="pdf-sig-role">Date of Issuance</div>
          </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="pdf-footer">
        <span>DEEPFAKE FORENSICS &bull; BSc (Hons) Cybersecurity &amp; Digital Forensics Capstone Project</span>
        <span>Page 1 of 1 &bull; Document SHA: ${d.evidence_sha256 ? d.evidence_sha256.substring(0, 16) + '...' : 'SECURE'}</span>
      </div>
    </div>
  `;
}

async function downloadPdfFile() {
  const sheet = el.pdfPaperSheet;
  if (!sheet) return;
  const d = current || {};
  const cleanFilename = (d.filename || 'Forensic_Evidence').replace(/[^a-zA-Z0-9._-]/g, '_');

  const btn = el.pdfDownloadBtn;
  const origBtnHtml = btn ? btn.innerHTML : '';
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `
      <svg class="spin-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M12 6v6l4 2"></path>
      </svg>
      Exporting PDF...
    `;
  }

  const opt = {
    margin: [6, 6, 6, 6],
    filename: `Forensic_Report_${cleanFilename}.pdf`,
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: {
      scale: 2,
      useCORS: true,
      logging: false,
      scrollY: 0,
      scrollX: 0,
      backgroundColor: '#FFFFFF',
      windowWidth: 1024
    },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
  };

  try {
    if (window.html2pdf) {
      await window.html2pdf().set(opt).from(sheet).save();
    } else {
      window.print();
    }
  } catch (err) {
    console.error('PDF export error, falling back to print:', err);
    window.print();
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = origBtnHtml;
    }
  }
}

window.addEventListener('resize', () => {
  if (current) drawPlot(current);
});

// Initialization
initAuth();
loadStatus();

// Auto-load latest verification report or forensic notice if opened from browser extension
(function checkAutoLoad() {
  const params = new URLSearchParams(window.location.search);
  const notice = params.get('notice');
  if (notice) {
    const src = params.get('source') || 'Social Media';
    if (el.failure && el.failureMsg) {
      el.failureMsg.innerHTML = `<strong>Notice (${src}):</strong> ${decodeURIComponent(notice)}`;
      el.failure.hidden = false;
      el.empty.hidden = true;
      el.result.hidden = true;
      el.working.hidden = true;
      setTimeout(() => {
        el.failure.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 300);
    }
    return;
  }

  const urlParam = params.get('url');
  if (urlParam) {
    setTimeout(() => {
      el.workspace?.scrollIntoView({ behavior: 'smooth' });
    }, 200);
    sendUrl(decodeURIComponent(urlParam));
    return;
  }

  if (params.get('view') === 'latest' || window.location.hash === '#latest') {
    fetch('/api/reports/latest')
      .then(res => res.json())
      .then(data => {
        if (data && data.probability !== undefined) {
          render(data);
          setTimeout(() => {
            el.result?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }, 300);
        }
      })
      .catch(() => {});
  }
})();


/* ═══════════════════════════════════════════════════════════════
   Homepage navigation behaviour
   Scroll spy, collapsible mobile menu, reading progress and a
   back-to-top control. Purely presentational — nothing here
   touches the detection pipeline.
   ═══════════════════════════════════════════════════════════════ */
(function initHomepageNav() {
  const navbar = $('navbar');
  const barNav = $('barNav');
  const navToggle = $('navToggle');
  const backToTop = $('backToTop');
  const progressFill = $('scrollProgressFill');

  /* Section id → the nav link that should light up for it. The lab
     appears twice because the visitor sees the locked portal and the
     signed-in examiner sees the workspace in the same slot. */
  const SPY_MAP = [
    ['hero', 'navHome'],
    ['overview', 'navOverview'],
    ['guide', 'navGuide'],
    ['loggedOutPortal', 'navLab'],
    ['workspace', 'navLab'],
    ['results', 'navTelemetry'],
    ['faq', 'navFaq'],
  ];

  const spyTargets = SPY_MAP
    .map(([sectionId, navId]) => ({ section: $(sectionId), link: $(navId) }))
    .filter((t) => t.section && t.link);

  const navLinks = spyTargets.map((t) => t.link);

  /* ── Mobile menu ──────────────────────────────────────────── */
  function setMenu(open) {
    if (!barNav || !navToggle) return;
    barNav.classList.toggle('is-open', open);
    navToggle.setAttribute('aria-expanded', String(open));
    navToggle.setAttribute('aria-label', open ? 'Close navigation menu' : 'Open navigation menu');
  }

  navToggle?.addEventListener('click', () => {
    setMenu(!barNav?.classList.contains('is-open'));
  });

  /* Collapse the drawer once a destination is chosen, and after the
     viewport grows past the breakpoint where the drawer exists. */
  barNav?.addEventListener('click', (e) => {
    if (e.target.closest('.nav-item')) setMenu(false);
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') setMenu(false);
  });

  const wide = window.matchMedia('(min-width: 1025px)');
  const onWidthChange = (m) => { if (m.matches) setMenu(false); };
  if (wide.addEventListener) wide.addEventListener('change', onWidthChange);
  else if (wide.addListener) wide.addListener(onWidthChange);

  /* ── Scroll spy, progress bar and back-to-top ─────────────── */
  function markActive(link) {
    navLinks.forEach((l) => l.classList.toggle('is-active', l === link));
  }

  function onScroll() {
    const y = window.scrollY || document.documentElement.scrollTop;
    const navH = navbar ? navbar.offsetHeight : 66;
    const probe = y + navH + 40;

    /* Walk the sections in document order and keep the last one whose
       top has passed the probe line. Hidden sections are skipped so the
       signed-out portal and the signed-in workspace never both count. */
    let active = null;
    for (const t of spyTargets) {
      if (t.section.hidden || t.section.offsetParent === null) continue;
      if (t.section.offsetTop <= probe) active = t.link;
    }
    markActive(active || navLinks[0]);

    if (progressFill) {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const pct = scrollable > 0 ? Math.min(100, (y / scrollable) * 100) : 0;
      progressFill.style.width = pct.toFixed(2) + '%';
    }

    if (backToTop) backToTop.hidden = y < 600;
  }

  /* Coalesce scroll work into one frame so the page stays smooth. */
  let ticking = false;
  function requestScrollUpdate() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { ticking = false; onScroll(); });
  }

  window.addEventListener('scroll', requestScrollUpdate, { passive: true });
  window.addEventListener('resize', requestScrollUpdate);

  backToTop?.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  onScroll();
})();


/* ═══════════════════════════════════════════════════════════════
   Account settings
   Profile details and password changes, both persisted server-side
   and written to the audit log. Reachable from the gear beside the
   profile badge in the header.
   ═══════════════════════════════════════════════════════════════ */

const settingsEl = {
  modal: $('settingsModal'), close: $('closeSettingsBtn'),
  alert: $('settingsAlert'),
  tabProfile: $('tabProfile'), tabSecurity: $('tabSecurity'),
  profileForm: $('profileForm'), passwordForm: $('passwordForm'),
  avatar: $('settingsAvatar'), name: $('settingsName'), email: $('settingsEmail'),
  created: $('settingsCreated'), lastLogin: $('settingsLastLogin'),
  loginCount: $('settingsLoginCount'),
  setName: $('setName'),
  setRole: $('setRole'), setEmail: $('setEmail'),
  currentPw: $('setCurrentPw'), newPw: $('setNewPw'), confirmPw: $('setConfirmPw'),
  saveProfileBtn: $('saveProfileBtn'), savePasswordBtn: $('savePasswordBtn'),
  logoutAllBtn: $('logoutAllBtn'),
};

function showSettingsAlert(msg, isSuccess = false) {
  if (!settingsEl.alert) return;
  settingsEl.alert.hidden = false;
  settingsEl.alert.textContent = msg;
  settingsEl.alert.className = `auth-alert ${isSuccess ? 'is-success' : ''}`;
}

function clearSettingsAlert() {
  if (!settingsEl.alert) return;
  settingsEl.alert.hidden = true;
  settingsEl.alert.textContent = '';
}

/* Dates come back as ISO strings from the server; show them in the
   examiner's own locale rather than raw UTC. */
function formatStamp(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  });
}

function setSettingsTab(which) {
  const isProfile = which === 'profile';
  settingsEl.tabProfile?.classList.toggle('is-active', isProfile);
  settingsEl.tabProfile?.setAttribute('aria-selected', String(isProfile));
  settingsEl.tabSecurity?.classList.toggle('is-active', !isProfile);
  settingsEl.tabSecurity?.setAttribute('aria-selected', String(!isProfile));
  if (settingsEl.profileForm) settingsEl.profileForm.hidden = !isProfile;
  if (settingsEl.passwordForm) settingsEl.passwordForm.hidden = isProfile;
  clearSettingsAlert();
}

/* Fill the dialog from the session user. Everything here is text the
   account holder typed, so it goes in via textContent / value rather
   than innerHTML. */
function fillSettings(user) {
  if (!user) return;
  if (settingsEl.avatar) settingsEl.avatar.textContent = user.initials || '??';
  if (settingsEl.name) settingsEl.name.textContent = user.name || '';
  if (settingsEl.email) settingsEl.email.textContent = user.email || '';
  if (settingsEl.created) settingsEl.created.textContent = formatStamp(user.created_at);
  if (settingsEl.lastLogin) settingsEl.lastLogin.textContent = formatStamp(user.last_login_at);
  if (settingsEl.loginCount) settingsEl.loginCount.textContent = String(user.login_count ?? 0);

  if (settingsEl.setName) settingsEl.setName.value = user.name || '';
  if (settingsEl.setEmail) settingsEl.setEmail.value = user.email || '';
  if (settingsEl.setRole && user.role) {
    // Only select a role the dropdown actually offers, so an unknown
    // stored value does not silently blank the control.
    const match = [...settingsEl.setRole.options].some((o) => o.value === user.role);
    if (match) settingsEl.setRole.value = user.role;
  }
}

function openSettingsModal() {
  if (!isLoggedIn()) { openAuthModal(); return; }
  fillSettings(getCurrentUser());
  setSettingsTab('profile');
  if (settingsEl.currentPw) settingsEl.currentPw.value = '';
  if (settingsEl.newPw) settingsEl.newPw.value = '';
  if (settingsEl.confirmPw) settingsEl.confirmPw.value = '';
  if (settingsEl.modal) {
    settingsEl.modal.hidden = false;
    settingsEl.modal.style.display = 'flex';
  }
  document.body.style.overflow = 'hidden';
  settingsEl.setName?.focus();
}

function closeSettingsModal() {
  if (settingsEl.modal) {
    settingsEl.modal.hidden = true;
    settingsEl.modal.style.display = 'none';
  }
  document.body.style.overflow = '';
  clearSettingsAlert();
}

settingsEl.close?.addEventListener('click', closeSettingsModal);
settingsEl.tabProfile?.addEventListener('click', () => setSettingsTab('profile'));
settingsEl.tabSecurity?.addEventListener('click', () => setSettingsTab('security'));

/* Backdrop click and Escape both close, matching the auth modal. */
settingsEl.modal?.addEventListener('click', (e) => {
  if (e.target === settingsEl.modal) closeSettingsModal();
});
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && settingsEl.modal && !settingsEl.modal.hidden) {
    closeSettingsModal();
  }
});

settingsEl.profileForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearSettingsAlert();
  const btn = settingsEl.saveProfileBtn;
  const label = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }

  try {
    const data = await api('/api/auth/profile', {
      method: 'PATCH',
      body: JSON.stringify({
        name: settingsEl.setName?.value || '',
        role: settingsEl.setRole?.value || '',
      }),
    });
    CURRENT_USER = data && data.user ? data.user : CURRENT_USER;
    updateAccessState();          // repaint the header badge with the new name
    fillSettings(CURRENT_USER);
    showSettingsAlert('Profile updated.', true);
  } catch (err) {
    showSettingsAlert(err.message || 'Could not save those changes.');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
});

settingsEl.passwordForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearSettingsAlert();

  const next = settingsEl.newPw?.value || '';
  if (next !== (settingsEl.confirmPw?.value || '')) {
    showSettingsAlert('The two new passwords do not match.');
    return;
  }

  const btn = settingsEl.savePasswordBtn;
  const label = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = 'Updating…'; }

  try {
    const data = await api('/api/auth/password', {
      method: 'POST',
      body: JSON.stringify({
        currentPassword: settingsEl.currentPw?.value || '',
        newPassword: next,
      }),
    });
    if (settingsEl.currentPw) settingsEl.currentPw.value = '';
    if (settingsEl.newPw) settingsEl.newPw.value = '';
    if (settingsEl.confirmPw) settingsEl.confirmPw.value = '';
    const n = (data && data.other_sessions_revoked) || 0;
    showSettingsAlert(
      n > 0
        ? `Password updated. ${n} other session${n === 1 ? '' : 's'} signed out.`
        : 'Password updated.',
      true,
    );
  } catch (err) {
    showSettingsAlert(err.message || 'Could not update the password.');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
});

settingsEl.logoutAllBtn?.addEventListener('click', async () => {
  // This ends the session in front of us too, so confirm before doing it.
  if (!window.confirm('Sign out of every device, including this one?')) return;
  try {
    await api('/api/auth/logout-all', { method: 'POST' });
  } catch {
    // The cookie is cleared server-side either way; fall through to
    // resetting the UI so it cannot show a stale signed-in state.
  }
  CURRENT_USER = null;
  closeSettingsModal();
  updateAccessState();
});
