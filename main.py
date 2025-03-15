import argparse
from torch.utils.data import DataLoader

from models import build_model
from data_loaders import Image_dataset

def create_loader(args, corrupt_flag, batch_size):
    assert args.num_images == args.batch_size

    dataset = Image_dataset(args, corrupt_flag)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True)

    return data_loader

parser = argparse.ArgumentParser(description='Network Details')


# General details
parser.add_argument('--model', default = 'DAM', type = str)
parser.add_argument('--pattern_size', default = 4096, type = int)
parser.add_argument('--training_epochs', default = 100, type = int)
parser.add_argument('--device', default = 'cuda:0')

# Data details
parser.add_argument('--folder_path', default = 'test_images\\cont', type = str)
parser.add_argument('--input_shape', default = 64)
parser.add_argument('--num_images', default = 4, type = int)
parser.add_argument('--batch_size', default = 4, type = int)
parser.add_argument('--pattern_type', default = 'continous')

# Evaluation
parser.add_argument('--perturb_percent', default = 0.2, type = float)
parser.add_argument('--crop_percent', default = 0.3, type = float)
parser.add_argument('--corrupt_type', default = 'both', type = str)
parser.add_argument('--evaluation_metric', default='hamming')
# parser.add_argument('--save_files', default = 'binary')

# Continous Hopfield
parser.add_argument('--mem_size', default = 4096, type = int)
parser.add_argument('--mem_dim', default = 4096, type = int)

args = parser.parse_args()

weight_folder = 'All_saves\\continous_DAM\\weights.pth'
visual_folder = 'O:\\PCodes\\Associative_memory\\DAM_experimentation_repo\\All_saves\\continous_DAM\\visual_saves'

model = build_model(args, weight_folder, visual_folder).to(args.device)
print(model)

training_loader = create_loader(args, corrupt_flag = False, batch_size = args.batch_size)
model.train(training_loader)
print('--- Done training ---')

testing_loader = create_loader(args, corrupt_flag = True, batch_size = 1)
model.recall(testing_loader)

