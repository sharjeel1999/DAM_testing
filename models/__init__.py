from classical_hopfield import Classical_HN


def build_model(model, args):
    if model == 'classical':
        return Classical_HN(args)