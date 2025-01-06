from models.classical_hopfield import Classical_HN


def build_model(args, weights_folder, visual_folder):
    if args.model == 'classical':
        return Classical_HN(args, weights_folder, visual_folder)
    elif args.model == 'DAM':
        return 0