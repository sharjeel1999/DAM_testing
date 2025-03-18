from models.classical_hopfield import Classical_HN
from models.DAM import Continous_DAM
from models.Spherical_memory import Spherical_memory

def build_model(args, weights_folder, visual_folder):
    if args.model == 'classical':
        return Classical_HN(args, weights_folder, visual_folder)
    elif args.model == 'DAM':
        return Continous_DAM(args, weights_folder, visual_folder)
    elif args.model == 'spherical':
        return Spherical_memory(args, weights_folder, visual_folder)