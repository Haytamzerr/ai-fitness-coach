// Coach Sportif Virtuel — real-time posture analysis in the browser.
// MoveNet (TensorFlow.js) finds 17 body keypoints in each webcam frame; joint angles drive a
// small UP/DOWN state machine that counts reps and gives spoken and written feedback.

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const videoWrap = document.getElementById('videoWrap');
const statusEl = document.getElementById('status');
const exerciseSelect = document.getElementById('exercise');
const startBtn = document.getElementById('startBtn');
const calibrateBtn = document.getElementById('calibrateBtn');
const repsEl = document.getElementById('reps');
const stateEl = document.getElementById('state');
const angleEl = document.getElementById('angle');
const feedbackEl = document.getElementById('feedback');
const audioCheckbox = document.getElementById('audioFeedback');
const mirrorCheckbox = document.getElementById('mirrorVideo');
const fpsInput = document.getElementById('fpsInput');
const fpsValue = document.getElementById('fpsValue');
const saveBtn = document.getElementById('saveBtn');
const resetBtn = document.getElementById('resetBtn');
const calMinEl = document.getElementById('calMin');
const calMaxEl = document.getElementById('calMax');
const historyEl = document.getElementById('history');

// ---------- settings ----------
const MIN_SCORE = 0.3;            // keypoints below this confidence are ignored
const SMOOTHING_SIZE = 6;         // frames in the moving average
const CALIBRATION_SAMPLES = 90;   // ~3 slow reps at 30 fps
const MIN_CALIBRATION_RANGE = 40; // degrees: below this the calibration is rejected
const CALIBRATION_MARGIN = 10;    // degrees kept inside the measured range
const SHALLOW_MARGIN = 20;        // a rep that went this far below "up" but not to "down"
const MAX_TORSO_LEAN = 50;        // degrees from vertical before warning on squats
const HISTORY_KEY = 'coach-history';

const DEFAULT_THRESHOLDS = {
  squat: { downAngle: 100, upAngle: 160 },   // knee angle (hip-knee-ankle)
  pushup: { downAngle: 90, upAngle: 155 },   // elbow angle (shoulder-elbow-wrist)
};
let thresholds = structuredClone(DEFAULT_THRESHOLDS);

// ---------- state ----------
let detector = null;
let running = false;
let reps = 0;
let state = 'UNKNOWN';
let smoothingWindow = [];
let lastFrameTime = 0;
let fpsCap = Number(fpsInput.value);
let sessionLog = [];
let sessionStart = null;
let repMinAngle = Infinity;        // lowest angle reached since the last time we were UP
let leftTop = false;               // true once the angle has dropped below the "up" threshold
let calibrating = false;
let calibrationSamples = [];
let lastSpoken = '';

// ---------- geometry ----------

// Angle ABC in degrees, at point B, between the segments BA and BC.
function angleBetween(a, b, c) {
  const abx = a.x - b.x, aby = a.y - b.y;
  const cbx = c.x - b.x, cby = c.y - b.y;
  const lengths = Math.hypot(abx, aby) * Math.hypot(cbx, cby);
  if (lengths === 0) return 180;
  const cos = Math.min(1, Math.max(-1, (abx * cbx + aby * cby) / lengths));
  return (Math.acos(cos) * 180) / Math.PI;
}

// Angle of the segment from `bottom` to `top` measured from the vertical, in degrees.
function angleFromVertical(top, bottom) {
  const dx = top.x - bottom.x, dy = bottom.y - top.y; // screen y grows downwards
  return Math.abs((Math.atan2(dx, dy) * 180) / Math.PI);
}

function midpoint(p, q) {
  return { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 };
}

function smooth(value) {
  smoothingWindow.push(value);
  if (smoothingWindow.length > SMOOTHING_SIZE) smoothingWindow.shift();
  return smoothingWindow.reduce((sum, v) => sum + v, 0) / smoothingWindow.length;
}

function visible(kp, ...names) {
  return names.every((n) => kp[n] && kp[n].score >= MIN_SCORE);
}

// ---------- camera and model ----------

async function setupCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Ce navigateur ne donne pas accès à la caméra.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function loadDetector() {
  await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
  await tf.ready();
  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
  });
}

// ---------- main loop ----------

async function runFrame(now) {
  if (!running) return;
  if (now - lastFrameTime >= 1000 / fpsCap) {
    lastFrameTime = now;
    const poses = await detector.estimatePoses(video, { maxPoses: 1, flipHorizontal: false });
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (poses.length) {
      const kp = {};
      for (const point of poses[0].keypoints) kp[point.name] = point;
      drawSkeleton(kp);
      handlePose(kp);
    } else {
      setFeedback('Personne détectée ? Recule un peu.', 'warn');
    }
  }
  requestAnimationFrame(runFrame);
}

const BONES = [
  ['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
];

function drawSkeleton(kp) {
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.9)';
  for (const [a, b] of BONES) {
    if (!visible(kp, a, b)) continue;
    ctx.beginPath();
    ctx.moveTo(kp[a].x, kp[a].y);
    ctx.lineTo(kp[b].x, kp[b].y);
    ctx.stroke();
  }
  ctx.fillStyle = '#f97316';
  for (const point of Object.values(kp)) {
    if (point.score < MIN_SCORE) continue;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// The joint angle that defines the movement, averaged over the sides the camera sees well.
function measureAngle(kp, exercise) {
  const joints = exercise === 'squat'
    ? [['left_hip', 'left_knee', 'left_ankle'], ['right_hip', 'right_knee', 'right_ankle']]
    : [['left_shoulder', 'left_elbow', 'left_wrist'], ['right_shoulder', 'right_elbow', 'right_wrist']];
  const angles = joints
    .filter((names) => visible(kp, ...names))
    .map(([a, b, c]) => angleBetween(kp[a], kp[b], kp[c]));
  if (!angles.length) return null;
  return angles.reduce((sum, v) => sum + v, 0) / angles.length;
}

function handlePose(kp) {
  const exercise = exerciseSelect.value;
  const raw = measureAngle(kp, exercise);
  if (raw === null) {
    setFeedback(exercise === 'squat' ? 'Jambes non visibles' : 'Bras non visibles', 'warn');
    return;
  }
  const angle = smooth(raw);
  angleEl.textContent = `${Math.round(angle)}°`;

  if (calibrating) {
    collectCalibration(angle);
    return;
  }

  const { downAngle, upAngle } = thresholds[exercise];
  const previous = state;
  repMinAngle = Math.min(repMinAngle, angle);

  if (angle <= downAngle) state = 'DOWN';
  else if (angle >= upAngle) state = 'UP';

  let feedback = '';
  let tone = 'ok';
  if (angle < upAngle) {
    leftTop = true; // the movement has started
  } else if (leftTop) {
    // back at the top: the movement is over, judge it
    if (previous === 'DOWN') {
      reps += 1;
      repsEl.textContent = reps;
      feedback = 'Bonne répétition !';
      announce(`Bonne répétition ! Total ${reps}`);
    } else if (repMinAngle < upAngle - SHALLOW_MARGIN) {
      // went down a bit but never reached the bottom: the rep does not count
      feedback = exercise === 'squat' ? 'Descends plus bas' : 'Fléchis plus les bras';
      tone = 'warn';
      announce(feedback);
    }
    leftTop = false;
    repMinAngle = Infinity;
  }

  if (exercise === 'squat' && state === 'DOWN' && visible(kp, 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')) {
    const shoulders = midpoint(kp.left_shoulder, kp.right_shoulder);
    const hips = midpoint(kp.left_hip, kp.right_hip);
    if (angleFromVertical(shoulders, hips) > MAX_TORSO_LEAN) {
      feedback = 'Redresse le buste';
      tone = 'warn';
      announce(feedback);
    }
  }

  stateEl.textContent = state === 'UP' ? 'Haut' : state === 'DOWN' ? 'Bas' : '—';
  if (feedback) setFeedback(feedback, tone);

  sessionLog.push({
    t: new Date().toISOString(),
    exercise,
    angle: Math.round(angle),
    state,
    feedback,
  });
}

function setFeedback(text, tone = 'ok') {
  feedbackEl.textContent = text;
  feedbackEl.className = tone;
}

// ---------- calibration ----------

function startCalibration() {
  if (!running) {
    setFeedback("Démarre d'abord la séance", 'warn');
    return;
  }
  calibrating = true;
  calibrationSamples = [];
  calibrateBtn.disabled = true;
  setFeedback('Calibration : 3 mouvements lents et complets', 'ok');
  announce('Calibration. Fais trois mouvements lents et complets.');
}

function collectCalibration(angle) {
  calibrationSamples.push(angle);
  if (calibrationSamples.length >= CALIBRATION_SAMPLES) finishCalibration();
}

function finishCalibration() {
  calibrating = false;
  calibrateBtn.disabled = false;
  const exercise = exerciseSelect.value;
  const minA = Math.min(...calibrationSamples);
  const maxA = Math.max(...calibrationSamples);
  calMinEl.textContent = `${Math.round(minA)}°`;
  calMaxEl.textContent = `${Math.round(maxA)}°`;
  if (maxA - minA < MIN_CALIBRATION_RANGE) {
    thresholds[exercise] = { ...DEFAULT_THRESHOLDS[exercise] };
    setFeedback('Amplitude trop faible : seuils par défaut gardés', 'warn');
    announce('Amplitude trop faible. Je garde les réglages par défaut.');
    return;
  }
  // The thresholds follow the person's own range of motion, in both directions.
  thresholds[exercise] = {
    downAngle: minA + CALIBRATION_MARGIN,
    upAngle: maxA - CALIBRATION_MARGIN,
  };
  state = 'UNKNOWN';
  repMinAngle = Infinity;
  leftTop = false;
  setFeedback('Calibration terminée', 'ok');
  announce('Calibration terminée.');
}

// ---------- voice ----------

function announce(text) {
  if (!audioCheckbox.checked || !('speechSynthesis' in window)) return;
  if (text === lastSpoken && speechSynthesis.speaking) return; // don't repeat over itself
  lastSpoken = text;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'fr-FR';
  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);
}

// ---------- session history (stays in this browser) ----------

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveSession() {
  if (!sessionStart || reps === 0) return;
  const history = loadHistory();
  history.unshift({
    date: sessionStart.toISOString(),
    exercise: exerciseSelect.value,
    reps,
    minutes: Math.max(1, Math.round((Date.now() - sessionStart) / 60000)),
  });
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 20)));
  } catch {
    /* storage full or blocked: history is a convenience */
  }
  renderHistory();
}

function renderHistory() {
  const history = loadHistory();
  historyEl.innerHTML = '';
  if (!history.length) {
    historyEl.innerHTML = '<li class="empty">Aucune séance enregistrée.</li>';
    return;
  }
  for (const s of history.slice(0, 8)) {
    const li = document.createElement('li');
    const day = new Date(s.date).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short' });
    const name = s.exercise === 'squat' ? 'Squats' : 'Pompes';
    li.innerHTML = `<span>${day} · ${name}</span><strong>${s.reps} reps</strong>`;
    historyEl.appendChild(li);
  }
}

// ---------- export and reset ----------

function exportCSV() {
  if (!sessionLog.length) {
    setFeedback('Rien à exporter pour le moment', 'warn');
    return;
  }
  const header = 'timestamp,exercise,angle,state,feedback';
  const rows = sessionLog.map((r) =>
    [r.t, r.exercise, r.angle, r.state, `"${r.feedback.replace(/"/g, '""')}"`].join(','));
  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `session_${Date.now()}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function resetSession() {
  reps = 0;
  state = 'UNKNOWN';
  smoothingWindow = [];
  sessionLog = [];
  repMinAngle = Infinity;
  leftTop = false;
  repsEl.textContent = '0';
  stateEl.textContent = '—';
  angleEl.textContent = '—';
  setFeedback('—');
}

function resetAll() {
  if (!confirm('Réinitialiser la séance et la calibration ?')) return;
  resetSession();
  thresholds = structuredClone(DEFAULT_THRESHOLDS);
  calMinEl.textContent = '—';
  calMaxEl.textContent = '—';
}

// ---------- events ----------

startBtn.addEventListener('click', () => {
  if (!detector) return;
  running = !running;
  startBtn.textContent = running ? 'Arrêter' : 'Démarrer';
  calibrateBtn.disabled = !running;
  if (running) {
    resetSession();
    sessionStart = new Date();
    announce('C\'est parti !');
    requestAnimationFrame(runFrame);
  } else {
    calibrating = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    saveSession();
    announce(`Séance terminée. ${reps} répétitions.`);
  }
});

exerciseSelect.addEventListener('change', resetSession);
calibrateBtn.addEventListener('click', startCalibration);
saveBtn.addEventListener('click', exportCSV);
resetBtn.addEventListener('click', resetAll);
fpsInput.addEventListener('input', () => {
  fpsCap = Number(fpsInput.value);
  fpsValue.textContent = fpsCap;
});
mirrorCheckbox.addEventListener('change', () => {
  videoWrap.classList.toggle('mirrored', mirrorCheckbox.checked);
});

// ---------- start-up ----------

(async function init() {
  renderHistory();
  try {
    await setupCamera();
    statusEl.textContent = 'Chargement du modèle…';
    await loadDetector();
    statusEl.textContent = '';
    startBtn.disabled = false;
    announce('Prêt. Appuie sur Démarrer pour commencer.');
  } catch (error) {
    console.error(error);
    statusEl.textContent = error.name === 'NotAllowedError'
      ? "Accès à la caméra refusé : autorise-le dans le navigateur."
      : `Erreur : ${error.message}`;
  }
})();
