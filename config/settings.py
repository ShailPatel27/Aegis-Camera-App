# Detection toggles (defaults)
DEFAULT_INTRUSION = True
DEFAULT_CROWD = True
DEFAULT_VEHICLE = False
DEFAULT_THREAT = False
DEFAULT_MOTION = False
DEFAULT_LOITER = False
DEFAULT_EMERGENCY = False

# YOLO
YOLO_MODEL_PATH = "models/yolov8n.pt"

# Performance
FRAME_INTERVAL_MS = 60  # ~16 FPS

# Motion Detection
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 1500

# Crowd Detection
CROWD_THRESHOLD = 3  # alert if >= this

# Logging
LOG_COOLDOWN = 2  # seconds