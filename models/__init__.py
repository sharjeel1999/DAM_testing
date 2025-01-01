from models.classical_hopfield import Classical_HN


def build_model(args):
    if args.model == 'classical':
        return Classical_HN(args)