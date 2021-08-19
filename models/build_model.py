from .featmodel import FeatModel, ImgModel
from .lstm import LSTMClassifier


def build_model(config):
    model_name = config.model
    model_type = config.type
    model_head = config.head

    if model_type == "feat":
        if model_head == "NN":
            i_c = config.input_channel
            a_s = config.align_size
            model = FeatModel(input_channel=i_c, align_size=a_s)

        elif model_head == "lstm":
            wm = config.width_mult
            ll = config.lstm_layers
            lh = config.lstm_hidden
            model = LSTMClassifier(wm, ll, lh)

    if model_type == "img":
        pass

    return model
