# Thresholds — tune these after observing real feature values in pipeline.py
EAR_SLEEPY      = 0.16   # eye aspect ratio below this → closing
PERCLOS_SLEEPY  = 0.40   # >45 % of rolling window with eyes closed
MAR_YAWN        = 0.65   # mouth aspect ratio above this → yawning
YAW_DISTRACTED  = 32.0   # degrees off-centre horizontally
PITCH_DISTRACTED = 24.0  # degrees off-centre vertically (absolute)

LABEL_NAMES = {0: 'ATTENTIVE', 1: 'SLEEPY', 2: 'DISTRACTED'}


class RuleBasedClassifier:
    """Deterministic threshold classifier. No training required."""

    def predict(self, features: dict) -> tuple:
        """Returns (label_str, label_int)."""
        ear     = features['ear_avg']
        perclos = features['perclos']
        mar     = features['mar']
        yaw     = abs(features['yaw'])
        pitch   = abs(features['pitch'])

        # Sleepy takes priority over distracted
        if ear < EAR_SLEEPY or perclos > PERCLOS_SLEEPY or mar > MAR_YAWN:
            return LABEL_NAMES[1], 1

        if yaw > YAW_DISTRACTED or pitch > PITCH_DISTRACTED:
            return LABEL_NAMES[2], 2

        return LABEL_NAMES[0], 0
