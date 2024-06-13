import torch
import pytorch_lightning as pl
from tqdm import tqdm
import time

from ..core.data import DataModule
from ..models.loader import get_model
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


class CustomSignLanguageDataset(Dataset):
    
    def __init__(self, input_pose, labels):
        
        self.input_pose = input_pose
        self.labels = labels
        
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        labels = self.labels[idx]
        input_pose = self.input_pose[idx]
        sample = {"Input_Pose": input_pose, "Glosses": torch.tensor(labels)}
        return sample
    
# merge with the corresponding modules in the future release.
class InferenceModel(pl.LightningModule):
    """
    This will be the general interface for running the inference across models.
    Args:
        cfg (dict): configuration set.

    """
    def __init__(self, cfg, stage="test"):
        super().__init__()
        self.cfg = cfg
        self.datamodule = DataModule(cfg.data)
        self.datamodule.setup(stage=stage)
        self.my_class = CustomSignLanguageDataset

        self.model = self.create_model(cfg.model)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if stage == "test":
            self.model.to(self._device).eval()
    
    def create_model(self, cfg):
        """
        Creates and returns the model object based on the config.
        """
        return get_model(cfg, self.datamodule.in_channels, self.datamodule.num_class)
    
    def forward(self, x):
        """
        Forward propagates the inputs and returns the model output.
        """
       
        return self.model(x)
    
    def init_from_checkpoint_if_available(self, map_location=torch.device("cpu")):
        """
        Intializes the pretrained weights if the ``cfg`` has ``pretrained`` parameter.
        """
        if "pretrained" not in self.cfg.keys():
            return

        ckpt_path = self.cfg["pretrained"]
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt
        
    
    
    def test_inference(self):
        """
        Calculates the time taken for inference for all the batches in the test dataloader.
        """
        
        print("Loading the corresponding pkl files")
        
        with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/targets_encoded.pkl", 'rb') as f:
            target_encoded = pickle.load(f)
            
        with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/original_inputs.pkl", 'rb') as f:
            original_input = pickle.load(f)
            
        with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/original_targets.pkl", 'rb') as f:
            original_target = pickle.load(f)
        with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/number_of_classes.pkl", 'rb') as f:
            clas = pickle.load(f)
            
        #print("Your encoder targets")
        #print(target_encoded)
                
        #print("Your Original targets")
        #print(original_target)
        
        #print("Your total number of classes")       
        #print(clas)
        
        output_sequence_length = max([len(p) for p in original_target])
        
        # TODO: Write output to a csv
        
        data_set = self.my_class(original_input,target_encoded)
        
        data_loader = DataLoader(data_set,batch_size = 59)
        
        
        #dataloader = self.datamodule.test_dataloader()
        total_time_taken, num_steps = 0.0, 0
        
        #print("Printing the data loader")
        #print(dataloader)
        
        
        #targets = torch.tensor(target_encoded)
                
        input_lengths = torch.full(
        size = (59,), 
        fill_value = 120,
        dtype=torch.int32
        )
                    
        #print("input_lengths",input_lengths)
                    
        target_lengths = torch.full(
        size = (59,), 
        fill_value = output_sequence_length,
        dtype=torch.int32
        )
                    
        #print("target_lengths",target_lengths)
        
        optimizer = torch.optim.Adam(self.model.parameters())
        
        
        n_epochs = 20
        
        
        for n in range(n_epochs):
        
            
        
        
            for batch in data_loader:
            
            
                optimizer.zero_grad()
            
            #print(batch.keys())
            #print(batch["frames"].shape)
            
            #with open('C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/input_batch_64'+'_'+str(i), 'wb') as handle:
                #pickle.dump(batch["frames"], handle)
            
            
            #print(batch["frames"])
            
                start_time = time.time()
                y_hat = self.model(batch["Input_Pose"].to(self._device)).cpu()
            
            #print(f'output shape after one batch {y_hat.shape}')
            
            #print("pick your input for continuous sign",y_hat.shape)
            
                y_hat = y_hat.permute(1,0,2)
            
            #print("After permutation",y_hat.size())
                log_soft_max_values =  torch.nn.functional.log_softmax(y_hat,2)
            
            
                        
            #targets = torch.randint(1, clas,(1,output_sequence_length))
            
            
            #print("Input_Log soft max",log_soft_max_values.size())
            #print("Target_length",targets.size())
        
            #print(f'Index{i}')
            
                loss = torch.nn.CTCLoss(blank=0)(
                    log_soft_max_values, batch["Glosses"], input_lengths, target_lengths
                )
            
                loss.backward()
            
                optimizer.step()
            
            
                
            print(f'Epoch {n} / Epochs {n_epochs} complited with loss = {loss.item()}')

            #class_indices = torch.argmax(y_hat, dim=-1)
            

            #for i, pred_index in enumerate(class_indices):
                
                #label = self.datamodule.test_dataset.id_to_gloss[int(pred_index.item())]
                #label = dataloader.dataset.id_to_gloss[int(pred_index)]
                #filename = batch["files"][i]

                #print("Hello , I have calculated")
                #print("pick your input for continuous sign",conti_x.shape)
                #print(f"{label}:\t{filename}")
            
            #total_time_taken += time.time() - start_time
            #num_steps += 1
        
        #print(f"Avg time per iteration: {total_time_taken*1000.0/num_steps} ms")

    def compute_test_accuracy(self):
        """
        Computes the accuracy for the test dataloader.
        """
        # Ensure labels are loaded
        assert not self.datamodule.test_dataset.inference_mode
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        dataset_scores, class_scores = {}, {}
        for batch_idx, batch in tqdm(enumerate(dataloader), unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)
            for i, (pred_index, gt_index) in enumerate(zip(class_indices, batch["labels"])):

                dataset_name = batch["dataset_names"][i]
                score = pred_index == gt_index
                
                if dataset_name not in dataset_scores:
                    dataset_scores[dataset_name] = []
                dataset_scores[dataset_name].append(score)

                if gt_index not in class_scores:
                    class_scores[gt_index] = []
                class_scores[gt_index].append(score)
        
        
        for dataset_name, score_array in dataset_scores.items():
            dataset_accuracy = sum(score_array)/len(score_array)
            print(f"Accuracy for {len(score_array)} samples in {dataset_name}: {dataset_accuracy*100}%")


        classwise_accuracies = {class_index: sum(scores)/len(scores) for class_index, scores in class_scores.items()}
        avg_classwise_accuracies = sum(classwise_accuracies.values()) / len(classwise_accuracies)

        print(f"Average of class-wise accuracies: {avg_classwise_accuracies*100}%")
    
    def compute_test_avg_class_accuracy(self):
        """
        Computes the accuracy for the test dataloader.
        """
        #Ensure labels are loaded
        assert not self.datamodule.test_dataset.inference_mode
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        scores = []
        all_class_indices=[]
        all_batch_labels=[]
        for batch_idx, batch in tqdm(enumerate(dataloader),unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)

            for i in range(len(batch["labels"])):
                all_batch_labels.append(batch["labels"][i])
                all_class_indices.append(class_indices[i])
            for pred_index, gt_index in zip(class_indices, batch["labels"]):
                scores.append(pred_index == gt_index)
        cm = confusion_matrix(np.array(all_batch_labels), np.array(all_class_indices))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Average Class Accuracy for {len(all_batch_labels)} samples: {np.mean(cm.diagonal())*100}%")
