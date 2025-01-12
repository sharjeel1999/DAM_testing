import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from typing import Optional, Tuple, Union

from DAM_core import HopfieldCore

class Continous_DAM():
    def __init__(self,
                 args,
                 weight_folder,
                 visual_folder,
                #  input_size: Optional[int] = None,
                #  hidden_size: Optional[int] = None,
                #  output_size: Optional[int] = None,
                #  pattern_size: Optional[int] = None,
                 num_heads: int = 1,
                 scaling: Optional[Union[float, Tensor]] = None,
                 update_steps_max: Optional[Union[int, Tensor]] = 0,
                 update_steps_eps: Union[float, Tensor] = 1e-4,

                 normalize_stored_pattern: bool = True,
                 normalize_stored_pattern_affine: bool = True,
                 normalize_stored_pattern_eps: float = 1e-5,
                 normalize_state_pattern: bool = True,
                 normalize_state_pattern_affine: bool = True,
                 normalize_state_pattern_eps: float = 1e-5,
                 normalize_pattern_projection: bool = True,
                 normalize_pattern_projection_affine: bool = True,
                 normalize_pattern_projection_eps: float = 1e-5,
                 normalize_hopfield_space: bool = False,
                 normalize_hopfield_space_affine: bool = False,
                 normalize_hopfield_space_eps: float = 1e-5,
                 stored_pattern_as_static: bool = False,
                 state_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 pattern_projection_as_connected: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,

                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False):
        
        super(Continous_DAM, self).__init__(args, weight_folder, visual_folder)

        input_size = args.input_shape
        hidden_size = args.pattern_size
        pattern_size = args.pattern_size
        output_size = args.pattern_size

        self.association_core = HopfieldCore(
            embed_dim=input_size, num_heads=num_heads, dropout=dropout, bias=input_bias,
            add_bias_kv=concat_bias_pattern, add_zero_attn=add_zero_association, kdim=stored_pattern_size,
            vdim=pattern_projection_size, head_dim=hidden_size, pattern_dim=pattern_size, out_dim=output_size,
            disable_out_projection=disable_out_projection, key_as_static=stored_pattern_as_static,
            query_as_static=state_pattern_as_static, value_as_static=pattern_projection_as_static,
            value_as_connected=pattern_projection_as_connected, normalize_pattern=normalize_hopfield_space,
            normalize_pattern_affine=normalize_hopfield_space_affine,
            normalize_pattern_eps=normalize_hopfield_space_eps)
        
        self.weights = torch.zeros((self.num_neurons, self.num_neurons))
        self.weights_transpose = self.weights.transpose(1, 0)

        self.optimizer = optim.Adam(self.weights, 0.001)
        self.loss_function = nn.HuberLoss(delta=1.0, reduction='mean')
        
    def train(self, pattern_loader):
        print('implement train')

        for pattern_dict in pattern_loader:
            pattern = torch.squeeze(pattern_dict['image'])
            associated_output = self.association_core(query = self.weights_transpose, key = pattern, value = self.weights)

            loss = self.loss_function(associated_output, pattern)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def recall(self, pattern_loader):
        print('implement recall')