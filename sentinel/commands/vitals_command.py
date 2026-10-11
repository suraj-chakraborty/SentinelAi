"""
sentinel/commands/vitals_command.py
────────────────────────────────────
Voice Command Plugin for Biometric Vitals & Wellness Dashboard.

Handles: "check my vitals", "wellness score", "posture check",
         "heart rate", "start health monitor", "I'm taking a break".
"""

import logging
from typing import Optional

from sentinel.perception.biometric_vitals import BiometricVitalsEngine, PostureState

logger = logging.getLogger("VitalsCommand")


class VitalsCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.engine = BiometricVitalsEngine()

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd = command.lower().strip()

        # 1. Start health monitoring
        if any(k in cmd for k in ("start health", "start vitals", "health monitor on",
                                   "biometric on", "monitor my health",
                                   "start biometric")):
            success = self.engine.start()
            if success:
                return "💓 Biometric vitals monitoring activated! Tracking heart rate, posture, eye strain, and stress levels via webcam."
            return "Health monitoring is already running."

        # 2. Stop monitoring
        if any(k in cmd for k in ("stop health", "stop vitals", "health monitor off",
                                   "biometric off", "stop biometric")):
            self.engine.stop()
            return "Biometric monitoring stopped. Session data saved."

        # 3. Current vitals
        if any(k in cmd for k in ("check vitals", "my vitals", "health status",
                                   "vital signs", "check my health")):
            v = self.engine.get_current_vitals()
            hr = f"{v['heart_rate_bpm']:.0f} BPM" if v['heart_rate_bpm'] else "Measuring..."
            stress = f"{v['stress_index']:.0%}" if v['stress_index'] is not None else "N/A"
            return (
                f"💓 Heart Rate: {hr}\n"
                f"👁️ Blink Rate: {v['blink_rate_per_min']:.1f}/min\n"
                f"🪑 Posture: {v['posture'].replace('_', ' ').title()}\n"
                f"😰 Stress Level: {stress}\n"
                f"🖥️ Screen Time: {v['screen_time_min']:.0f} min\n"
                f"⏰ Last Break: {v['last_break_min_ago']:.0f} min ago"
            )

        # 4. Wellness score
        if any(k in cmd for k in ("wellness score", "health score", "how healthy",
                                   "wellness report", "daily wellness")):
            score = self.engine.compute_wellness_score()
            grade = "🟢" if score.overall >= 80 else "🟡" if score.overall >= 60 else "🔴"
            return (
                f"{grade} Wellness Score: {score.overall:.0f}/100\n"
                f"  💓 Heart Health: {score.heart_health:.0f}/100\n"
                f"  👁️ Eye Health: {score.eye_health:.0f}/100\n"
                f"  🪑 Posture: {score.posture_score:.0f}/100\n"
                f"  😌 Stress: {score.stress_score:.0f}/100\n"
                f"  ⏰ Break Compliance: {score.break_compliance:.0f}/100"
            )

        # 5. Heart rate
        if any(k in cmd for k in ("heart rate", "pulse", "my heart")):
            hr = self.engine._hr_estimator._last_hr
            if hr:
                return f"💓 Your current estimated heart rate is {hr:.0f} BPM."
            return "Heart rate not available yet. Make sure the webcam can see your face."

        # 6. Posture check
        if any(k in cmd for k in ("posture check", "am i slouching", "posture status",
                                   "check posture")):
            posture = self.engine._posture_analyzer.current_posture
            msgs = {
                PostureState.UPRIGHT: "✅ Your posture looks great! Keep it up.",
                PostureState.SLIGHT_LEAN: "⚠️ Slight lean detected. Try sitting a bit more upright.",
                PostureState.SLOUCHING: "🔴 You're slouching! Sit up straight and align your monitor at eye level.",
                PostureState.UNKNOWN: "Posture not measured yet. Enable biometric monitoring first.",
            }
            return msgs.get(posture, "Posture status unknown.")

        # 7. Take a break
        if any(k in cmd for k in ("taking a break", "i'm taking a break", "break time",
                                   "took a break", "just took a break")):
            return self.engine.record_break()

        # 8. Stress level
        if any(k in cmd for k in ("stress level", "how stressed", "am i stressed")):
            stress = self.engine._hr_estimator.get_stress_index()
            if stress is not None:
                level = "calm" if stress < 0.3 else "moderate" if stress < 0.6 else "high"
                return f"😌 Stress Level: {stress:.0%} ({level}). {'Take some deep breaths!' if stress > 0.5 else 'Looking good!'}"
            return "Stress level not available yet."

        return "Vitals command not recognized. Try: check my vitals, wellness score, posture check, start health monitor."
