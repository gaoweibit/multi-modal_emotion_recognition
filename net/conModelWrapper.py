import os
import sys
from . import fusion_model
from transformers import Wav2Vec2Model
import torch
from torch import nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.getcwd())
import utils

class ConModelWrapper():
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = args.device
        self.model_type = args.model_type
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.output_num = args.output_num
        self.lab_type = args.label_type
        self.lbl_learning  = args.label_learning
        self.lr = args.lr
        self.model_path = args.model_path


    def init_model(self):
        """
        Define model and load pretrained weights
        """
        assert self.model_type in [
            "wav2vec2-large", "wav2vec2-large-robust"], \
            print("Wrong model type")
        
        default_models={
            "wav2vec2": "wav2vec2-large-robust",
        }
        real_model_name = default_models.get(self.model_type, self.model_type)
        assert real_model_name not in ["wav2vec2"], \
            print("Model name is not properly converted.\n \
                Current model_name:", real_model_name
            )
        
        root_model_type = real_model_name.split("-")[0]
        assert root_model_type in ["wav2vec2"], \
            print("Can't specify the root model type\n \
                Current root_model_type:", root_model_type
            )

        arch_type = real_model_name.split("-")[1]
        assert arch_type in ["base", "large"], \
            print("Can't specify the architecture type\n \
                architecture_type:", arch_type
            )

        # If base model, set is_large to False
        if arch_type == "large":
            is_large = True 
        elif arch_type == "base":
            is_large = False 
        else: 
            raise ValueError
        print("Loading", real_model_name)

        #### Wav2vec2
        if root_model_type == "wav2vec2":
            self.wav2vec_model= Wav2Vec2Model.from_pretrained(os.path.join(os.path.dirname(__file__), 'wav2vec2'))
            self.wav2vec_model.freeze_feature_encoder()
            del self.wav2vec_model.encoder.layers[12:]
            
        idim = 1024 if is_large else 768

        self.fusion_model = fusion_model.FusionModel(self.args)

        self.wav2vec_model.to(self.device)

        self.fusion_model.to(self.device)

        self.model_type_list = ["head", "wav2vec"]

        
    def init_optimizer(self):
        """
        Define optimizer for pre-trained model
        """

        assert self.wav2vec_model is not None and self.fusion_model is not None, \
            print("Model is not initialized")
        
        self.wav2vec_opt = optim.Adam(self.wav2vec_model.parameters(), lr=self.lr)
        self.fusion_model_opt = optim.Adam(self.fusion_model.parameters(), lr=self.lr)
        self.scaler = GradScaler()
    
    def feed_forward(self, xa, xv, eval=False, **kwargs):
        """
        Feed forward the model
        """
        def __inference__(self, xa, xv, **kwargs):
            mask = kwargs.get("attention_mask", None)
            if self.model_type == "wav2vec1":
                z = self.wav2vec_model.feature_extractor(xa)
                w2v = self.wav2vec_model.feature_aggregator(z)
            else:
                with torch.no_grad():
                    w2v = self.wav2vec_model(xa, attention_mask=mask).last_hidden_state

            pred, loss1, loss2 = self.fusion_model(w2v,xv)

            return pred, loss1, loss2
        
        if eval:
            with torch.no_grad():
                return __inference__(self, xa, xv, **kwargs)
        else:
            return __inference__(self, xa, xv, **kwargs)
    
    def backprop(self, total_loss):
        """
        Update the model given loss
        """
        self.fusion_model_opt.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.fusion_model_opt)
        self.scaler.update()

    def save_model(self, epoch):
        """
        Save the model for each epoch
        """
  
        torch.save(self.wav2vec_model.state_dict(), \
            os.path.join(self.model_path, 'final_model.pt'))
        torch.save(self.fusion_model.state_dict(), \
            os.path.join(self.model_path, "final_head.pt"))
    
    def save_final_model(self, min_epoch, remove_param=False):
        """
        Copy the given epoch model to the final model
            if remove_param is True, remove the param folder
        """
        
        os.system("cp "+os.path.join(self.model_path, "param", str(min_epoch)+"_head.pt") + \
        " "+os.path.join(self.model_path, "final_head.pt"))
        os.system("cp "+os.path.join(self.model_path, "param", str(min_epoch)+"_model.pt") + \
            " "+os.path.join(self.model_path, "final_wav2vec.pt"))

        if remove_param:
            os.system("rm -rf "+os.path.join(self.model_path, "param"))

    def set_eval(self):
        """
        Set the model to eval mode
        """
        self.wav2vec_model.eval()
        self.fusion_model.eval()
    def set_train(self):
        """
        Set the model to train mode
        """
        self.wav2vec_model.eval()
        self.fusion_model.train()

    def load_model(self, model_path, run_type):
        if run_type == 'train':
            self.wav2vec_model.load_state_dict(torch.load(model_path+"/final_wav2vec.pt"))
        else:
            self.wav2vec_model.load_state_dict(torch.load(model_path+"/final_model.pt"))
            self.fusion_model.load_state_dict(torch.load(model_path+"/final_head.pt"))

