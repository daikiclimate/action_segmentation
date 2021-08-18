from .featmodel import FeatModel, ImgModel


def build_model(config):
    model_name = config.model
    model_type = config.type
    if model_type == "feat":
        i_c = config.input_channel
        a_s = config.align_size
        model = FeatModel(input_channel=i_c, align_size=a_s)

    if model_type == "img":
        pass

    return model
