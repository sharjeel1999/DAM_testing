import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from typing import Optional, Tuple, Union

from models.DAM_core import HopfieldCore
from models.Hopfield_core import Hopfield_Core

class Continous_DAM(Hopfield_Core):
    def __init__(self,
                 args,
                 weight_folder,
                 visual_folder,
                #  input_size: Optional[int] = None,
                #  hidden_size: Optional[int] = None,
                #  output_size: Optional[int] = None,
                #  pattern_size: Optional[int] = None,
                #  num_heads: int = 1,
                #  scaling: Optional[Union[float, Tensor]] = None,
                #  update_steps_max: Optional[Union[int, Tensor]] = 0,
                #  update_steps_eps: Union[float, Tensor] = 1e-4,

                #  normalize_stored_pattern: bool = True,
                #  normalize_stored_pattern_affine: bool = True,
                #  normalize_stored_pattern_eps: float = 1e-5,
                #  normalize_state_pattern: bool = True,
                #  normalize_state_pattern_affine: bool = True,
                #  normalize_state_pattern_eps: float = 1e-5,
                #  normalize_pattern_projection: bool = True,
                #  normalize_pattern_projection_affine: bool = True,
                #  normalize_pattern_projection_eps: float = 1e-5,
                #  normalize_hopfield_space: bool = False,
                #  normalize_hopfield_space_affine: bool = False,
                #  normalize_hopfield_space_eps: float = 1e-5,
                #  stored_pattern_as_static: bool = False,
                #  state_pattern_as_static: bool = False,
                #  pattern_projection_as_static: bool = False,
                #  pattern_projection_as_connected: bool = False,
                #  stored_pattern_size: Optional[int] = None,
                #  pattern_projection_size: Optional[int] = None,

                #  batch_first: bool = True,
                #  association_activation: Optional[str] = None,
                #  dropout: float = 0.0,
                #  input_bias: bool = True,
                #  concat_bias_pattern: bool = False,
                #  add_zero_association: bool = False,
                #  disable_out_projection: bool = False
                 ):
        
        super(Continous_DAM, self).__init__(args, weight_folder, visual_folder)

        # input_size = args.pattern_size
        # hidden_size = args.pattern_size
        self.pattern_size = args.pattern_size
        self.mem_size = args.mem_size

        # self.association_core = HopfieldCore(
        #     embed_dim=input_size, num_heads=num_heads, dropout=dropout, bias=input_bias,
        #     add_bias_kv=concat_bias_pattern, add_zero_attn=add_zero_association, kdim=stored_pattern_size,
        #     vdim=pattern_projection_size, head_dim=hidden_size, pattern_dim=pattern_size, out_dim=output_size,
        #     disable_out_projection=disable_out_projection, key_as_static=stored_pattern_as_static,
        #     query_as_static=state_pattern_as_static, value_as_static=pattern_projection_as_static,
        #     value_as_connected=pattern_projection_as_connected, normalize_pattern=normalize_hopfield_space,
        #     normalize_pattern_affine=normalize_hopfield_space_affine,
        #     normalize_pattern_eps=normalize_hopfield_space_eps)
        
        self.weights = nn.Parameter(torch.rand((self.mem_size, self.mem_size)))
        # self.weights_transpose = self.weights.transpose(1, 0)

        self.query_proj = nn.Linear(self.pattern_size, self.mem_size)
        self.key_proj = nn.Linear(self.pattern_size, self.mem_size)
        self.value_proj = nn.Linear(self.pattern_size, self.mem_size)
        self.output_proj = nn.Linear(self.mem_size, self.pattern_size)


        self.optimizer = optim.Adam([self.weights], 0.001)
        self.loss_function = nn.HuberLoss(delta=1.0, reduction='mean')
        
    def association_forward(self, pattern):
        print('--- pattern data type:' , pattern.dtype)
        q = self.query_proj(pattern)
        k = self.key_proj(self.weights)
        v = self.value_proj(self.weights)

        attn_weights = F.softmax(torch.matmul(q, k.t()) / (self.mem_size ** 0.5), dim = -1)
        attn_output = torch.matmul(attn_weights, v)

        output = self.output_proj(attn_output)
        return output

    def train(self, pattern_loader):
        print('implement train')

        for pattern_dict in pattern_loader:
            pattern = torch.squeeze(pattern_dict['image']).float()
            # print('===== Raw q/k/v inpu shapes:', pattern.shape, self.weights.shape)
            # associated_output = self.association_core(query = self.weights_transpose, key = pattern, value = self.weights)
            associated_output = self.association_forward(pattern)
            # print('associated shape: ', associated_output.shape)
            loss = self.loss_function(associated_output, pattern)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.save_weights()

    def recall(self, pattern_loader, steps=5):
        print('implement recall')

        self.load_weights()
        weights_transpose = self.weights.transpose(1, 0)

        for m, pattern_dict in enumerate(pattern_loader):
            print('index: ', m)
            pattern = torch.squeeze(pattern_dict['image']).float()
            perturbed_pattern = torch.squeeze(pattern_dict['perturbed']).float()
            print('Pattern shapes: ', pattern.shape, perturbed_pattern.shape)
        
            pattern = torch.tensor(pattern, dtype=torch.float32)
            perturbed_pattern = torch.tensor(perturbed_pattern, dtype=torch.float32)
            copied_pattern = pattern.clone()
            perturbed_hamming = self.calculate_similarity(perturbed_pattern, pattern)
            print(f'Perturbed Hamming Score: {perturbed_hamming}')

            print(f'Recovering pattern for {steps} steps.')
            for s in range(steps):
                perturbed_pattern = perturbed_pattern.unsqueeze(dim = 0)
                # perturbed_pattern = self.association_core(query = weights_transpose, key = perturbed_pattern, value = self.weights)
                perturbed_pattern = self.association_forward(perturbed_pattern)
                perturbed_pattern = torch.squeeze(perturbed_pattern)
                hamming = self.calculate_similarity(perturbed_pattern, pattern)
                print(f'Step: {s}, Hamming Score: {hamming}')

            self.save_files(pattern, perturbed_pattern, m)

