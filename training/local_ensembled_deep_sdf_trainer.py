import torch

from training import BaseTrainer
from utils.eval import compute_evaluation_metrics, make_grid_points
from utils.loss_functions import igr_loss


class LocalEnsembledDeepSDFTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, test_loader, 
                 latent_optimizer, latent_scheduler, latent_codes, cfg):
        super(LocalEnsembledDeepSDFTrainer, self).__init__(model, optimizer, scheduler, train_loader, val_loader, test_loader, 
                                                          latent_optimizer, latent_scheduler, latent_codes, cfg)

        self.alpha = cfg['loss']['kwargs']['alpha']
        self.epochs_til_evaluation = self.cfg['train']['epochs_til_evaluation']

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        self.latent_optimizer.zero_grad()

        loss_dict = self.compute_loss(batch_data)
      
        total_loss = 0
        for weight, loss in zip(self.cfg['loss']['weights'], list(loss_dict.values())):
            total_loss += weight * loss

        total_loss.backward()  

        if 'gradient_clip' in self.cfg['misc'] and self.cfg['misc']['gradient_clip'] > 0:
            max_norm = self.cfg['misc']['gradient_clip']
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_norm)
            # torch.nn.utils.clip_grad_norm_(self.latent_codes.parameters(), max_norm = max_norm) I don't know why, but this doesn't work.
        
            # Clip gradients of the latent codes using custom implementation.
            for param in self.latent_codes.parameters():
                if param.grad is not None:
                    grad = param.grad.coalesce()   
                    norm = grad._values().norm(2) 
                    if norm > max_norm:
                        scale = max_norm / (norm + 1e-6)
                        grad._values().mul_(scale)

        log_dict = {
            'loss_dict': {f'train/{key}': value for key, value in loss_dict.items()},
            'train/total_loss': total_loss
        }

        self.to_train_log(log_dict)

        self.optimizer.step()
        self.latent_optimizer.step()

    def compute_loss(self, batch_data):     
        shape_ids = batch_data['shape_id']

        surf_points = batch_data['surf_points'].to(self.device)
        surf_normals = batch_data['surf_normals'].to(self.device)
        landmarks = batch_data['landmarks'].to(self.device)
        vol_points_near = batch_data['vol_points_near'].to(self.device)
        vol_points_far = batch_data['vol_points_far'].to(self.device)

        points = torch.cat([surf_points, vol_points_near, vol_points_far], dim = 1)
        points = points.requires_grad_()

        latent_code = self.latent_codes(shape_ids).to(self.device)
        latent_code = latent_code.unsqueeze(1).repeat(1, points.shape[1], 1) 
      
        pred_sdf, pred_anchors = self.model(points, latent_code)
      
        loss_dict = igr_loss(pred_sdf, points, surf_normals, latent_code, self.alpha)
        loss_dict['anchor_loss'] = (torch.norm(pred_anchors - landmarks, dim = -1)).mean()
       
        return_dict = {}
        for objective in self.cfg['loss']['objectives']:
            return_dict[objective] = loss_dict[objective]

        return return_dict

    def eval_step(self, batch_data):
        voxel_resolution = self.cfg['eval']['voxel_resolution']
        chunk_size = self.cfg['eval']['chunk_size']

        shape_ids = batch_data['shape_id']

        # Points for evaluation.
        chamfer_points = batch_data['chamfer_points']
        chamfer_normals = batch_data['chamfer_normals']

        grid_points = make_grid_points(voxel_resolution).repeat(chamfer_points.shape[0], 1, 1).to(self.device) 
        chunks = torch.split(grid_points, chunk_size, dim = 1)

        fx_chunks = []
        for points in chunks:
            latent_code = self.latent_codes(shape_ids).to(self.device)
            latent_code = latent_code.unsqueeze(1).repeat(1, points.shape[1], 1)        
            
            fx, _ = self.model(points, latent_code)
            fx = fx.squeeze(dim = -1).detach().cpu()
            fx_chunks.append(fx)
        fx = torch.cat(fx_chunks, dim = 1)

        fx_volume = fx.reshape(fx.shape[0], voxel_resolution, voxel_resolution, voxel_resolution)

        mesh, metrics = compute_evaluation_metrics(fx_volume, chamfer_points, chamfer_normals, shape_ids)

        return mesh, metrics