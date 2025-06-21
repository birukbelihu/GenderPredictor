def get_app_name() -> str:
    return "Gender Predictor"

def get_face_detector_prototext_file() -> str:
    return "models/deploy.prototxt"


def get_face_detector_caffe_model() -> str:
    return "models/face_detector.caffemodel"


def get_gender_predictor_model() -> str:
    return "models/gender_predictor.h5"


def get_gender_predictor_model_classes() -> list[str]:
    return ["Female", "Male"]


def exit_keys() -> tuple:
    return 'q', 'Q', 'e', 'E'
