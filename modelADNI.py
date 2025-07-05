import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Sequence

#from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
#from monai.networks.blocks.transformerblock import TransformerBlock

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

from typing import Union #stk

#from modeling_m3d_clip import ViT

import copy

from ResNetmodel import generate_model

from hpt.utils.utils import get_sinusoid_encoding_table


STD_SCALE = 0.02 # from HPT

class MLPDecoder(nn.Module):
    def __init__(self, hidden_size = 768):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        #print("fc input:", x.shape) #stk
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class StateEncoder(nn.Module):
    """
    Reference from hpt/models/policy_stem.py MLP class
    """
    def __init__(self, input_dim=280, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        

    def forward(self, x):
        #print("fc input:", x.shape) #stk
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
 
        return x

    
class ADNIModel(nn.Module):
    
    def __init__(
        self,
        trunk,
        image_stem,  # from HPT model.  # take input of batch size x num tokens x 512
        image_encoder_depth=18,
        image_encoder_pretrained_path=None,
        share_image_encoder=True,
        state_input_dim=280,
        modality_embed_dim=256,
        modality_names_types={"sag": "image",
                              "cor": "image",
                              "axi": "image",
                              "volume": "state"},
        use_modality_tokens=False,  # whether to use a modality token to represent each modality
        #modality_token_nums={"image":16, "state": 16},
        batch_normalization=True,
    ):
        super().__init__()
        
        self.trunk = trunk # the shared embedding space
        self.modality_embed_dim = modality_embed_dim
        
        ##############################################################
        # initiate the image encoder
        self.share_image_encoder = share_image_encoder
        
        resnet_backbone = generate_model(model_depth=image_encoder_depth, 
                       n_classes=1039
                      )   # takes input: batch size x Channel (3 for this pre-trained model) x D x H x W
        if image_encoder_pretrained_path is not None:
            checkpoint = torch.load(image_encoder_pretrained_path, map_location='cpu')
            message = resnet_backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Load image encoder pretrained model:", message)

        self.image_encoder = nn.Sequential(*list(resnet_backbone.children())[:-2]) # output: batch size x 512 x D' x H' x W'
        
        # turn off the batch normalization:
        def remove_batchnorm_layers(model):
            # Recursively replace BatchNorm3d with nn.Identity in all submodules
            for name, module in model.named_children():
                if isinstance(module, torch.nn.BatchNorm3d):
                    setattr(model, name, torch.nn.Identity())
                else:
                    remove_batchnorm_layers(module)  # Recursively apply to nested modules
                    
        if not batch_normalization:
            remove_batchnorm_layers(self.image_encoder)
        ##############################################################
        
        ##############################################################
        # initiate the state encoder
        self.state_input_dim = state_input_dim
        self.state_encoder = StateEncoder(input_dim=self.state_input_dim, output_dim=self.modality_embed_dim) # generate one single token for volume vec
        ##############################################################
        
        # multi-modality stems:
        self.modality_names_types = modality_names_types
        self.use_modality_tokens = use_modality_tokens
        
        assert self.use_modality_tokens == False, "Currently not support using modality tokens!" #stk: TODO
        
        self.modality_stems = {}
        
        if self.use_modality_tokens:
            self.modality_tokens = {} # represent the type of modality
        
        if self.share_image_encoder == False:
            self.modality_encoders = {}
        
        self.init_stem(image_stem)
        
        self.modality_stems = nn.ModuleDict(self.modality_stems)
        if self.use_modality_tokens:
            self.modality_tokens = nn.ModuleDict(self.modality_tokens)
        if self.share_image_encoder == False:
            self.modality_encoders = nn.ModuleDict(self.modality_encoders)
            del self.image_encoder
        
        
        self.head = MLPDecoder(hidden_size=self.modality_embed_dim)
        

    def init_stem(self, image_stem):
        """
        image_stem: take input of batch size x num tokens x 512
        Get the stem for each modality
        """

        for modality_name in self.modality_names_types.keys():
            modality_type = self.modality_names_types[modality_name]
            
            if self.use_modality_tokens:
                self.modality_tokens[modality_name] = nn.Parameter(torch.randn(1, 1, self.modality_embed_dim) * STD_SCALE)
            
            if "image" in modality_type:
                self.modality_stems[modality_name] = copy.deepcopy(image_stem)
                if self.share_image_encoder == False:
                    self.modality_encoders[modality_name] = copy.deepcopy(self.image_encoder)

    
    def stem_process(self, data):
        features = []
        for modality in data.keys():
            val = data[modality]
            modality_type = self.modality_names_types[modality]
            
            if "image" in modality_type:
                # run the image encoder / patchifier
                image_encoder = self.image_encoder if self.share_image_encoder == True else self.modality_encoders[modality]
                
                patched_image = image_encoder(val) # val: batch x 3 x D x H x W, output: batch x 512 x D' x H' X W'
                patched_image  = patched_image.permute([0, 2, 3, 4, 1]) # batch x D' x H' x W' x 512
                
                # add pos embedding
                data_shape = patched_image.shape
                positional_embedding = get_sinusoid_encoding_table(
                    0, int(torch.prod(torch.tensor(data_shape[1:-1]))), data_shape[-1]
                ).to(patched_image.device) # shape: 1 x num tokens (k,v) x 512
                    
                positional_embedding = positional_embedding.repeat([data_shape[0], 1, 1])
                patched_image = patched_image + positional_embedding.view(patched_image.shape)
            
            
                # send into stem
                curr_stem_token = self.modality_stems[modality].compute_latent(patched_image) # input: batch x D' x H' x W' x embed dim; output: batch x num tokens (q) x embed dim
                
                if self.use_modality_tokens:
                    assert self.use_modality_tokens == False, "Currently not support using modality tokens!" # TODO
                    
                    
            
            if "state" in modality_type:
                curr_stem_token = self.state_encoder(val).unsqueeze(1) # add the token dim. shape: batch x 1 x embed_dim
                
                if self.use_modality_tokens:
                    assert self.use_modality_tokens == False, "Currently not support using modality tokens!" # TODO
                
            features.append(curr_stem_token)
            
        return torch.cat(features, dim=1) # batch x num tokens (all q) x embed dim
                
            
    def preprocess_token(self, tokens):
        # add position embedding:
        
        positional_embedding = get_sinusoid_encoding_table(
                    0, int(tokens.shape[1]), tokens.shape[-1]
                ).to(tokens.device) # shape: 1 x num tokens x embed dim
                    
        positional_embedding = positional_embedding.repeat([tokens.shape[0], 1, 1])
        tokens = tokens + positional_embedding
        return tokens
        
    def forward(self, data):
        """
        data: dict. {"sag": x1, "cor": x2, "axi": x3, "volume": x4}
        """
        
        stem_tokens = self.stem_process(data)
        
        stem_tokens = self.preprocess_token(stem_tokens) # add pos encoding
        
        trunk_tokens = self.trunk(stem_tokens) # batch size x num tokens x embed dim
        result = self.head(trunk_tokens.mean(dim=1))
        
        return result