"""Scientific parameters and output protocol identifiers."""


PAPER_VERSION = "paper-mainline-v1"

DECISION_PROTOCOL = "intersection-location-v1"


PREPROCESSING = "full_series_minmax_then_split"

WINDOW_LOWER_BOUND = 10
NUM_WINDOW_CANDIDATES = 4

DEFAULT_ACF_FALLBACK = 80

ACF_MIN_LAG = 10
ACF_MAX_LAG = 1000
ACF_PEAK_DISTANCE = 10


N_KERNEL = 1000

KSS_SAMPLES = 100
KSS_KEEP_RATE = 0.5

KSS_ALPHA = 1.0
KSS_BETA = 1.0

K_NEIGHBORS = 3


SECOND_VIEW = "window_curvature"

CURVATURE_PADDING = "reflect"


TRAIN_KNN = "leave_one_out"
TEST_REFERENCE = "train_only"

TAIL_Q = 0.95


VIEW_FUSION = "colocal_a3"

LOCAL_RADIUS = "source_window"


CANDIDATE_REGION = "test_complete_window_endpoints"
CANDIDATE_PROTOCOL = "test-complete-window-endpoint-v1"
TOP_K_PER_WINDOW = 1

EVENT_INTERVAL = "end_aligned_[p-w+1,p]"

EVENT_GROUPING = "common_intersection"
EVENT_SCORE = "sum"


EVENT_LOCALIZATION = "common_intersection_right_endpoint"


PAPER_SEEDS = (0, 1, 2)
