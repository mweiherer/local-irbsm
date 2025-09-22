import torch.nn as nn
import torch
import numpy as np

from utils.common import count_parameters


# Code adapted from https://github.com/SimonGiebenhain/NPHM/blob/main/src/NPHM/models/EnsembledDeepSDF.py.
class EnsembledLinear(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features, bias = True):
        super(EnsembledLinear, self).__init__()
        self.ensemble_size = ensemble_size
   
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight = torch.nn.Parameter(torch.Tensor(ensemble_size, out_features, in_features))
      
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else: 
            self.register_parameter('bias', None)

        # Initialize weights and biases.
        for i in range(self.ensemble_size):
            torch.nn.init.kaiming_uniform_(self.weight[i, ...], a = np.sqrt(5))

            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[i, ...])
                bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias[i, ...], -bound, bound)

    def forward(self, input):
        output = torch.bmm(self.weight, input.permute(0, 2, 1)).permute(0, 2, 1)

        if self.bias is not None:
            output += self.bias.unsqueeze(dim = 1)
     
        return output


# Code adapted from https://github.com/SimonGiebenhain/NPHM/blob/main/src/NPHM/models/EnsembledDeepSDF.py.
class EnsembledDeepSDF(nn.Module):
    def __init__(self, ensemble_size, input_dim, output_dim, latent_dim, hidden_dim, num_layers):   
        super(EnsembledDeepSDF, self).__init__()
        d_in = input_dim + latent_dim

        dims = [hidden_dim] * num_layers
        dims = [d_in] + dims + [output_dim]

        self.num_layers = len(dims)
        self.skip_in = [num_layers // 2]

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = EnsembledLinear(ensemble_size, dims[layer], out_dim)

            setattr(self, "lin" + str(layer), lin)

        self.actvn = nn.Softplus(beta = 100)

    def forward(self, input_points, latent_code):
        A, B, N, _ = input_points.shape

        input = torch.cat([input_points, latent_code], dim = -1).reshape(A, B * N, -1)
        x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.actvn(x)

        return x.reshape(A, B, N, -1)


# Code adapted from https://github.com/SimonGiebenhain/NPHM/blob/main/src/NPHM/models/EnsembledDeepSDF.py.
class LocalEnsembledDeepSDF(nn.Module):
    def __init__(self, cfg):
        super(LocalEnsembledDeepSDF, self).__init__()
        ensemble_cfg = cfg['model']['local_ensembled_deep_sdf']

        anchor_indices = ensemble_cfg['anchor_indices']
        num_anchors = len(anchor_indices)
   
        input_dim = ensemble_cfg['input_dim']
        output_dim = ensemble_cfg['output_dim']
        global_latent_dim = ensemble_cfg['global_latent_dim']
        local_latent_dim = ensemble_cfg['local_latent_dim']
        hidden_dim = ensemble_cfg['hidden_dim']
        num_layers = ensemble_cfg['num_layers']

        anchor_mlp_hidden_dim = cfg['model']['anchor_mlp']['hidden_dim']

        self.ensembled_deep_sdf = EnsembledDeepSDF(ensemble_size = num_anchors + 1,
                                                   input_dim = input_dim,
                                                   output_dim = output_dim,
                                                   latent_dim = global_latent_dim + local_latent_dim,
                                                   hidden_dim = hidden_dim,
                                                   num_layers = num_layers)
        
        print('Number of trainable parameters (without anchor MLP):', count_parameters(self))

        self.anchor_mlp = nn.Sequential(
            nn.Linear(global_latent_dim, anchor_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(anchor_mlp_hidden_dim, anchor_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(anchor_mlp_hidden_dim, 3 * num_anchors))

        self.num_anchors = num_anchors
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
       
        self.sigma2 = ensemble_cfg['blending']['sigma'] ** 2
        self.const = ensemble_cfg['blending']['const']
  
        # Note: The average anchors are already in [-1, 1]^3.
        self.average_anchors = torch.load(cfg['data']['average_anchors'])[anchor_indices, ...]
        
        print('Number of trainable parameters (with anchor MLP):', count_parameters(self))
    
    def blend_ensembled_sdf(self, q, p, f, background = True):
        '''
        Blends multiple local SDFs into a single SDF using a Gaussian kernel. 
        :param q: The query points as torch.Tensor of size [b, n, 3]
        :param p: The anchor points as torch.Tensor of size [b, num_anchors, 3]
        :param f: The SDF values as torch.Tensor of size [b, n, num_anchors + 1, 1]
        :param background: If True, a constant value for the background model is added
        :return: The blended SDF values as torch.Tensor of size [b, n, 1]
        '''
        dist = -((p.unsqueeze(dim = 1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(dim = 2)).norm(dim = 3) + 1e-5) ** 2

        if background:
            dist_const = torch.ones_like(dist[:, :, :1]) * -self.const
            dist = torch.cat([dist, dist_const], dim = -1)

        weight = (dist / self.sigma2).exp()
        weight = weight / (weight.sum(dim = 2).unsqueeze(dim = -1) + 1e-6)

        return (weight.unsqueeze(dim = -1) * f).sum(dim = 2) 
 
    def forward(self, input_points, latent_code):
        '''
        Forward method of ensembled DeepSDF model.
        :param input_points: The input points as torch.Tensor of size [b, n, 3]
        :param latent_code: The latent code as torch.Tensor 
            of size [b, n, global_latent_dim + (num_anchors + 1) * local_latent_dim]
        :return: The predicted SDF values as torch.Tensor of size [b, n, 1] 
            and anchor points as torch.Tensor of size [b, num_anchors, 3]
        '''
        B, N, _ = input_points.shape
  
        # Predict anchor positions as offsets to average anchors.
        pred_anchors = self.anchor_mlp(latent_code[:, 0, :self.global_latent_dim] ).reshape(-1, self.num_anchors, 3)
        pred_anchors += self.average_anchors.to(pred_anchors)
        
        # Represent points in local coordinate system; use global coordinate system for last anchor point.
        _pred_anchors = pred_anchors.unsqueeze(dim = 1).repeat(1, N, 1, 1)
        xyz = (input_points.unsqueeze(dim = 2) - 
               torch.cat([_pred_anchors, torch.zeros_like(_pred_anchors[:, :, :1, :])], dim = 2)).permute(2, 0, 1, 3)

        # Prepare latent codes for ensemble.
        global_latent = latent_code[:, :, :self.global_latent_dim].unsqueeze(dim = 2).repeat(1, 1, self.num_anchors + 1, 1)
        local_latent = latent_code[:, :, self.global_latent_dim:].reshape(B, -1, self.num_anchors + 1, self.local_latent_dim)
        latents = torch.cat([global_latent, local_latent], dim = -1).permute(2, 0, 1, 3)
      
        pred_sdf = self.ensembled_deep_sdf(xyz, latents).permute(1, 2, 0, 3)

        # Blend predictions into a single SDF.
        pred_sdf = self.blend_ensembled_sdf(input_points, pred_anchors, pred_sdf)

        return pred_sdf, pred_anchors