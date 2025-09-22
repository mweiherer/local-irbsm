import torch
import torchvision

from datasets import BreastDataset
from datasets.transforms import ShapeReconstruction

from models import LocalEnsembledDeepSDF
from training import LocalEnsembledDeepSDFTrainer

from utils.common import seed_everything
from utils.io import dtype_from_string


dataset_dict = {
    'breast': BreastDataset
}

module_dict = {
    'local_ensembled_deep_sdf': (LocalEnsembledDeepSDF, LocalEnsembledDeepSDFTrainer)
}


def make_dataset(cfg):
    dataset = cfg['data']['dataset']

    dataset_transform = torchvision.transforms.Compose([ShapeReconstruction(cfg)])

    DatasetClass = dataset_dict[dataset]
    
    train_dataset = DatasetClass(cfg, mode = 'train', transform = dataset_transform)
    val_dataset = DatasetClass(cfg, mode = 'val', transform = dataset_transform)  
    test_dataset = DatasetClass(cfg, mode = 'test', transform = dataset_transform)  

    return train_dataset, val_dataset, test_dataset

def make_dataloaders(cfg):
    train_dataset, val_dataset, test_dataset = make_dataset(cfg)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = cfg['train']['batch_size'],
                                               shuffle = True,
                                               num_workers = cfg['train']['num_workers'],
                                               pin_memory = True,
                                               worker_init_fn = 
                                                    lambda w_id: seed_everything(cfg['misc']['manual_seed'] + w_id))
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size = 1,
                                             shuffle = False,
                                             pin_memory = True,
                                             num_workers = 1,
                                             worker_init_fn = 
                                                    lambda w_id: seed_everything(cfg['misc']['manual_seed'] + w_id))
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size = 1,
                                              shuffle = False,
                                              pin_memory = True,
                                              num_workers = 1,
                                              worker_init_fn = 
                                                    lambda w_id: seed_everything(cfg['misc']['manual_seed'] + w_id))

    return train_loader, val_loader, test_loader

def make_model(cfg, checkpoint = None):
    model = module_dict[cfg['module']][0](cfg)

    dtype = dtype_from_string(cfg['misc']['dtype'])
    model = model.to(dtype)

    if checkpoint is not None:
        model.load_state_dict(checkpoint)

    return model

def make_optimizer(cfg, model):
    method = cfg['optimizer']['method']

    if method == 'adam':
        return torch.optim.Adam(model.parameters(),
                                lr = cfg['optimizer']['lr'],
                                betas = cfg['optimizer']['optimizer_kwargs']['betas'])
    if method == 'adam_w':
        return torch.optim.AdamW(list(model.parameters()),
                                 lr = cfg['optimizer']['lr'],
                                 weight_decay = cfg['optimizer']['optimizer_kwargs']['weight_decay'])
    
    raise ValueError(f"'{method}' is not a supported optimizer at the moment. Choose either 'adam' or 'adam_w'.")

def make_latent_optimizer(cfg, latent_codes):
    method = cfg['latent_optimizer']['method']

    if method == 'adam':
        return torch.optim.Adam(list(latent_codes.parameters()),
                                lr = cfg['latent_optimizer']['lr'],
                                betas = cfg['latent_optimizer']['optimizer_kwargs']['betas'])
    
    if method == 'adam_w':
        return torch.optim.SparseAdam(list(latent_codes.parameters()),
                                      lr = cfg['latent_optimizer']['lr'])
    
    raise ValueError(f"'{method}' is not a supported latent code optimizer at the moment. Choose 'adam' or 'adam_w'.")

def make_scheduler(cfg, optimizer):
    method = cfg['optimizer']['lr_scheduler']

    if method == 'step_lr':
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size = cfg['optimizer']['scheduler_kwargs']['step_size'],
                                               gamma = cfg['optimizer']['scheduler_kwargs']['gamma'])
    
    raise ValueError(f"{method} is not a supported scheduler at the moment. Choose 'step_lr'.")

def make_latent_scheduler(cfg, latent_optimizer):
    method = cfg['latent_optimizer']['lr_scheduler']

    if method == 'step_lr':
        return torch.optim.lr_scheduler.StepLR(latent_optimizer,
                                               step_size = cfg['latent_optimizer']['scheduler_kwargs']['step_size'],
                                               gamma = cfg['latent_optimizer']['scheduler_kwargs']['gamma'])
    
    raise ValueError(f"{method} is not a supported latent code scheduler at the moment. Choose 'step_lr'.")

def make_latent_codes(cfg, dataset_size):
    ensemble_cfg = cfg['model']['local_ensembled_deep_sdf']
    num_anchors = len(ensemble_cfg['anchor_indices'])
    latent_dim = ensemble_cfg['global_latent_dim'] + (num_anchors + 1) * ensemble_cfg['local_latent_dim']
    
    if 'sparse_embedding' in cfg['misc'] and cfg['misc']['sparse_embedding']:
        latent_codes = torch.nn.Embedding(dataset_size, latent_dim, max_norm = 1.0, sparse = True)
    else:
        latent_codes = torch.nn.Embedding(dataset_size, latent_dim)

    torch.nn.init.normal_(latent_codes.weight.data, std = cfg['misc']['latent_init'])
    
    return latent_codes

def make_trainer(cfg):
    '''
    Setup training from given config file.
    :param cfg: The config file
    :return trainer: The trainer that is responsible for training the given model
    '''
    model = make_model(cfg) 
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    latent_codes = make_latent_codes(cfg, len(train_loader.dataset)) # Latent codes of training dataset.
    latent_optimizer = make_latent_optimizer(cfg, latent_codes)
    latent_scheduler = make_latent_scheduler(cfg, latent_optimizer)

    trainer = module_dict[cfg['module']][1](model, optimizer, scheduler, train_loader, val_loader, test_loader, 
                                            latent_optimizer, latent_scheduler, latent_codes, cfg)

    return trainer