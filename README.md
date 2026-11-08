# AI Fitness Coach — Coach Sportif Virtuel

A virtual fitness coach that runs entirely in the browser. It watches you through the webcam, finds your body's joints in real time, counts your **squats** and **push-ups**, and tells you (out loud) when a rep is good or when your form needs fixing.

No server, no install, no account: the video never leaves your computer.

> Interface in French. Built as a university project at UEMF (supervised by Mr. Abderrahim Waga), December 2025.

## Features

- **Real-time pose detection** with MoveNet (TensorFlow.js), drawn as a skeleton over the video
- **Automatic rep counting** for squats (knee angle) and push-ups (elbow angle)
- **Form feedback**, written and spoken (Web Speech API):
  - *"Descends plus bas"* — you came back up without reaching the bottom (the rep is not counted)
  - *"Redresse le buste"* — your torso leans too far forward during a squat
- **Calibration** to your own range of motion (3 slow reps), so the counter works for different bodies and camera angles
- **Session history** saved in the browser (last 20 sessions)
- **CSV export** of every frame (time, exercise, angle, state, feedback) for later analysis
- Mirror mode, FPS cap, voice on/off

## How it works

```
webcam frame ──► MoveNet ──► 17 keypoints ──► joint angle ──► smoothing ──► state machine ──► reps + feedback
```

1. **Keypoints.** MoveNet SinglePose Lightning returns 17 points (shoulders, elbows, hips, knees, ankles…) with a confidence score. Points under 0.3 are ignored, so a half-hidden body doesn't produce fake reps.
2. **Joint angle.** The angle at the knee (hip–knee–ankle) or the elbow (shoulder–elbow–wrist) is computed with the dot product:
   `θ = arccos( (BA · BC) / (|BA| |BC|) )`, averaged over the sides the camera sees.
3. **Smoothing.** A moving average over the last 6 frames removes the jitter of the detector.
4. **State machine.** Below the *down* threshold the state is `DOWN`, above the *up* threshold it is `UP`. A rep is counted when the movement goes `UP → DOWN → UP`. If you come back up without reaching `DOWN`, the rep is rejected and you hear why.
5. **Calibration.** The coach records your lowest and highest angle over ~3 seconds and sets the thresholds 10° inside that range. If the range is too small (under 40°), it keeps the defaults instead of learning a broken setting.

| Exercise | Joint | Default *down* | Default *up* |
|---|---|---|---|
| Squat | knee | 100° | 160° |
| Push-up | elbow | 90° | 155° |

## Run it

The camera only works on `https://` or `localhost`, so serve the folder instead of double-clicking the file:

```bash
git clone https://github.com/Haytamzerr/ai-fitness-coach.git
cd ai-fitness-coach
python -m http.server 8000
```

Then open <http://localhost:8000> in Chrome or Edge, allow the camera, and:

1. choose the exercise, click **Démarrer**;
2. (optional) click **Calibrer** and do 3 slow, full reps;
3. train; click **Arrêter** to save the session, **Exporter CSV** to download the data.

Tip: stand 2–3 m from the camera so your whole body is in the frame, with even lighting. For push-ups, film from the side.

## Tech stack

- HTML, CSS, vanilla JavaScript
- [TensorFlow.js](https://www.tensorflow.org/js) 4.12 + [pose-detection](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) (MoveNet)
- Web APIs: `getUserMedia`, Canvas 2D, `speechSynthesis`, `localStorage`

## Project structure

```
index.html   page layout (video, controls, stats)
style.css    styles, responsive under 980 px
script.js    camera, model, angles, rep counter, calibration, voice, history, export
```

## Roadmap

- More exercises (lunges, plank hold timer, jumping jacks)
- A small classifier trained on recorded sessions to recognise form errors
- Progress charts from the session history
- Kalman filter instead of the moving average for smoother angles
