CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")
HISTORY_FRAMES = tuple(range(0, 10))
FUTURE_FRAMES = tuple(range(10, 90))
MAP_CHANNELS = ("drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "road_divider", "lane_divider", "road_block")
MODEL_SLUGS = ("horizondrive", "magicdrive", "magicdrivedit", "magicdrivev2", "opendwm", "panacea", "basedreamer", "dreamforge", "dreamforge_dit")
PILOT_INDICES = (0, 250, 325, 425)
IMAGE_SIZE = (400, 224)
SCHEMA_VERSION = "genbench-gt2-prediction-v1"
