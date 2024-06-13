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

from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.model_selection import train_test_split


from pprint import pprint

#from ctc_decoder import best_path, beam_search

from torchmetrics import WordErrorRate

import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2




class CustomSignLanguageDataset_train(Dataset):
    
    def __init__(self, input_pose, labels, orig_train_targets):
        
        self.input_pose = input_pose
        self.labels = labels
        self.orig_train_targets = orig_train_targets
        
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        labels = self.labels[idx]
        input_pose = self.input_pose[idx]
        orig_train_targets = self.orig_train_targets[idx]
        sample1 = {"Input_Pose": input_pose, "Glosses": torch.tensor(labels), "original_train_targets": orig_train_targets}
        return sample1
    
    
class CustomSignLanguageDataset_validation(Dataset):
    
    def __init__(self, input_pose, labels, orig_val_targets):
        
        self.input_pose = input_pose
        self.labels = labels
        self.orig_val_targets = orig_val_targets
        
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        labels = self.labels[idx]
        input_pose = self.input_pose[idx]
        orig_val_targets = self.orig_val_targets[idx]
        sample2 = {"Input_Pose": input_pose, "Glosses": torch.tensor(labels), "original_val_targets": orig_val_targets}
        return sample2
    
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
        
        
        ###########################################################################
        
        # Defining Training data loader class
        self.my_train_class = CustomSignLanguageDataset_train
        
        
        # Defining Validation data loader class
        self.my_val_class = CustomSignLanguageDataset_validation
        
        ###########################################################################
        
        

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
        
    # Number classses + 1 for blank class
    
    def test_inference(self):
        """
        Calculates the time taken for inference for all the batches in the test dataloader.
        """
        

        
        
       
        #print("Loading the corresponding pkl files")
        
        #with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/targets_encoded.pkl", 'rb') as f:
            #target_encoded = pickle.load(f)
            
         
        
    ######################################################   Input Output labels #############################################################
            
        with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/train_inputs.pkl", 'rb') as f:  #1
            train_original_input = pickle.load(f)
            
        with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/train_targets.pkl", 'rb') as f:  #2
            train_original_target = pickle.load(f)
            
       # with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/train_targets_encoded_added_one.pkl", 'rb') as f:
           # train_encoded_target = pickle.load(f)
            
        with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/test_input.pkl", 'rb') as f:  #3
            validation_original_input = pickle.load(f)
            
        with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/test_targets.pkl", 'rb') as f:  #4
            validatioin_original_target = pickle.load(f)
            
        #with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/validation_encoded_added_one.pkl", 'rb') as f:
            #validatioin_encoded_target = pickle.load(f)
    ##############################################################################################################################################        
            
        #with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/updated_original_targets.pkl", 'rb') as f:
           # orig_targets_for_validation = pickle.load(f)
        #with open("C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/number_of_classes.pkl", 'rb') as f:
            #clas = pickle.load(f)
            
        #print("Your encoder targets")
        #print(target_encoded)
                
        #print("Your Original targets")
        #print(original_target)
        
        #print("Your total number of classes")       
        #print(clas)
        
        ###################################################   Data preprocessing  ##################################################################################
        
        print("Started Pre-Processing your data--------------------------------")
        
        
        
        
        #print("Initial data")
        #print(orig_targets_for_validation)
        
        max_len = max([len(l) for l in train_original_target])
        
        for x11 in train_original_target:
            
            #z89 = x11[-1]
            
    
            y11 = ['Epsilon']*(max_len-len(x11))
    
            x11.extend(y11)
        
        #print("After Padded")
        #print(original_target)
        
        targets_flat = [c for clist in train_original_target for c in clist]
        label_enc = LabelEncoder()
        
        label_enc.fit(targets_flat)
        
        # Getting encoded output

        train_targets_enc = [label_enc.transform(x) for x in train_original_target]

        # as zero we are keeping for unknown , we need to add one to the lsit below
        train_targets_enc = np.array(train_targets_enc)+1
        
        clas = len(label_enc.classes_)
        
        print("Total number of classes")
        print(clas)
        
        ######################################################################################################
        
        max_len_val = max([len(l) for l in validatioin_original_target])
        
        for x12 in validatioin_original_target:
            
            #z80 = x12[-1]
            
    
            y12 = ['Epsilon']*(max_len_val-len(x12))
    
            x12.extend(y12)
        
        #print("After Padded")
        #print(original_target)
        
        targets_flat = [c for clist in train_original_target for c in clist]
        #label_enc = LabelEncoder()
        
        label_enc.fit(targets_flat)
        
        # Getting encoded output

        validation_targets_enc = [label_enc.transform(x) for x in validatioin_original_target]

        # as zero we are keeping for unknown , we need to add one to the lsit below
        validation_targets_enc = np.array(validation_targets_enc)+1
        
        
        
        
        #################   Train Test split ##########################
        
       # train_input, val_input, train_label, val_label, train_orig_targets, val_orig_targets = train_test_split(original_input,targets_enc, original_target, test_size=0.152)
        
       # print(f"Train input shape is {train_input.shape}")
        
       # print(f"Test input shape is {val_input.shape}")
        
        
        #print("Validation original is")
        #print(val_orig_targets)
        
        train_batch_size = 25
        
        val_batch_size = 25
        
      #######################################################################################################################################
        
        
        output_sequence_length = max([len(p) for p in train_targets_enc])
        
        #print(output_sequence_length)
        
        # TODO: Write output to a csv
        
        #########################################################################
        
        #####################  Data Loader ######################################
        
        train_data_set = self.my_train_class(train_original_input,train_targets_enc,train_original_target)
        
        val_data_set = self.my_val_class(validation_original_input,validation_targets_enc,validatioin_original_target)
        
       
        
        train_data_loader = DataLoader(train_data_set,batch_size = train_batch_size, shuffle = True)
        val_data_loader = DataLoader(val_data_set, batch_size = val_batch_size, shuffle = False)

        
            
            
        
        print("Your data pre-processing finished -------------------------------------------------------")
        
        #######################################################################
        
        
        
        

        
        #print("Printing the data loader")
        #print(dataloader)
        
        ############################################################################################################
        ####################################            BERT              #########################################
        
        
        # For BERT model input sequence time stamp is 120
        
        
        #targets = torch.tensor(target_encoded)
                
        train_input_lengths_bert = torch.full(size = (train_batch_size, ), fill_value = 120, dtype=torch.int32)
                    
        #print("input_lengths",input_lengths)
                    
        train_target_lengths_bert = torch.full(size = (train_batch_size, ), fill_value = output_sequence_length, dtype=torch.int32)
                    
        #print("target_lengths",target_lengths)
        
        
        # For BERT model input sequence time stamp is 120
                
        val_input_lengths_bert = torch.full(size = (val_batch_size,), fill_value = 120, dtype=torch.int32)
                    
        #print("input_lengths",input_lengths)
                    
        val_target_lengths_bert = torch.full(size = (val_batch_size,), fill_value = output_sequence_length, dtype=torch.int32)
                    
        #print("target_lengths",target_lengths)
        
        # optimizer for BERT
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4, weight_decay = 0.001)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8, verbose = True)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, last_epoch = -1, verbose = True)
        
        # Multiplicative LR
        
        #lmbda = lambda epoch: 0.98
        
        #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=100, momentum = 0.9)
        
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        
        
        #targets = torch.tensor(target_encoded)
        
        ###########################################################################################################
        

        
        
       
        ############################################################################################################
        ####################################            BiLSTM                #########################################
        
        # For RNN model input sequence time stamp is 91
                
        #train_input_lengths_lstm = torch.full(size = (train_batch_size, ), fill_value = 120, dtype=torch.int32)
                    
        #print("input_lengths",input_lengths)
                    
        #train_target_lengths_lstm = torch.full(size = (train_batch_size, ), fill_value = output_sequence_length, dtype=torch.int32)
                    
        #print("target_lengths",target_lengths)
        
        
        # For RNN model input sequence time stamp is 91
                
       # val_input_lengths_lstm = torch.full(size = (val_batch_size,), fill_value = 120, dtype=torch.int32)
                    
        #print("input_lengths",input_lengths)
                    
        #val_target_lengths_lstm = torch.full(size = (val_batch_size,), fill_value = output_sequence_length, dtype=torch.int32)
                    
        #print("target_lengths",target_lengths)
        
        # optimizer for biLSTM
       # optimizer = torch.optim.Adam(self.model.parameters(), lr = 7e-4)
       # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8, verbose = True)
        
        #torch.backends.cudnn.enabled=False
        
        ##################################################################################################################
       
        
        n_epochs = 6000
        
        train_loss_list = []
        
        val_loss_list = []
        
        word_error_rate_epoch_list = []
        
        
        
       ######################################################## pickel file generation for "pass" statement and Batch label pkl file genearation ################# 
        
        #dataloader = self.datamodule.test_dataloader()
        #total_time_taken, num_steps = 0.0, 0
        #i = 0
        #for batch in dataloader:
            
            #pass
            #print(i+1)
            #print(batch.keys())
            #print(batch["frames"].shape)
            
            #for j in range(len(batch["files"])):
                
                #filename = batch["files"][j]
                
                #print(f"{filename}")

               # print("Hello , I have calculated")
               # print("pick your input for continuous sign",conti_x.shape)
            #print("Completed")
            #with open('D:/Datasets for sign language/Complete video dataset for indian sign language/ICSL_25/Complete_file/sample'+'_'+str(i)+'.pkl', 'wb') as handle:
                #pickle.dump(batch["frames"], handle)
                
            #i = i + 1
            
            
        
        
        
        
        
    
        for n in range(n_epochs):
        
          
            
            
          ############################################################################################################
            
            ############################         BiLSTM   and BERT   (Training)   #########################################################  
            
            i = 0
            p = 0
            train_batch_loss = 0
            
            val_batch_loss = 0
            
            orig_preds = []
            
            
            
            wer = WordErrorRate()
            
        
            for train_batch in train_data_loader:
                      
                #print(batch.keys())
                #print(batch["frames"].shape)
                
                #with open('C:/Users/aswin/OneDrive/Desktop/temp/OpenHands/openhands/apis/input_batch_64'+'_'+str(i), 'wb') as handle:
                    #pickle.dump(batch["frames"], handle)
                
                
            #print(batch["frames"])
            
            
                start_time = time.time()
                #y_hat = self.model(batch["Input_Pose"].to(self._device)).cpu()
                
                #print("Batch input size", batch["frames"].shape)
                
                
                
                optimizer.zero_grad()
                y_hat = self.model(train_batch["Input_Pose"].to(self._device)).cpu()
                
                y_hat = y_hat.permute(1,0,2)
                
                log_soft_max_values =  torch.nn.functional.log_softmax(y_hat,2)
                
                #targets = torch.randint(1, clas,(1,output_sequence_length))
                
                train_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(log_soft_max_values, train_batch["Glosses"], train_input_lengths_bert, train_target_lengths_bert)
                
                #print("Training loss is here")
                #print(train_loss.item())
                
                train_batch_loss = train_batch_loss + train_loss.item()
                
                train_loss.backward()
                
                optimizer.step()
                
                i = i + 1
            
                
                
                
            ###############################################################################################################
            
            
            
            ############################################################################################################
            
            ############################         BiLSTM   and BERT   (Validation)   #########################################################
            
            
            
            
            
            for val_batch in  val_data_loader:
                
                
                y_val_hat = self.model(val_batch["Input_Pose"].to(self._device)).cpu()
                
                
                y_val_hat = y_val_hat.permute(1,0,2)
                
                log_soft_max_values_val =  torch.nn.functional.log_softmax(y_val_hat,2)
                
                #targets = torch.randint(1, clas,(1,output_sequence_length))
                
                #print(log_soft_max_values_val.shape)
                
                val_loss = torch.nn.CTCLoss(blank=0)(log_soft_max_values_val, val_batch["Glosses"], val_input_lengths_bert, val_target_lengths_bert)
                
                
                #print("Validation loss and p is here")
                #print(val_loss.item())
                #print(p)
                
                val_batch_loss = val_batch_loss + val_loss.item()
                
                p = p + 1
                
                ###############################   validation accuracy   #################################
                
                y_val_hat = y_val_hat.permute(1,0,2)
                
                y_val_hat = torch.softmax(y_val_hat,2)
                
                y_val_hat = torch.argmax(y_val_hat,2)
                
                with torch.no_grad():
                    
                    y_val_hat = y_val_hat.detach().cpu().numpy()
                    
                    cap_preds = []
                    
                    total_batch_preds = []
                    
                    #print(y_val_hat.shape)
                    
                    for k in range(y_val_hat.shape[0]):
                        
                        
                        temp = []
                        
                        for m in y_val_hat[k,:]:  
                            
                            
                            
                            m = m - 1  # Substracting 1 as we have added in label encoder case
                            
                            if m == -1:
                                
                                temp.append("$")
                                
                            else:
                                
                                temp.append(label_enc.inverse_transform([m])[0])
                                
                                
                        tp = " ".join(temp)
                        
                        cap_preds.append(tp)
                        
                        
                        
                        
                        
                    total_batch_preds.extend(cap_preds)
                    
                     
                        
                
                
            ##################################################   Word Error Rate Calculation   #######################################################        
                    
            print("Two samples of Validation Original Padded Ground Truth and Predicted Output")
            pprint(list(zip(validatioin_original_target, total_batch_preds))[6:8])
            
            
            word_error_rate_list = []
            
            word_error_rate = torch.zeros((val_batch_size))
            
            for s1 in range(val_batch_size):
                
                    
                 val_orig_targets_new = [x for x in validatioin_original_target[s1] if x != "Epsilon"]
                    
                 a = ' '.join(val_orig_targets_new)
                 
                 #print("Converted Input string")
                 #print(a)
                 
                 
                 new_pred_list = ["#"]
                 
                 
                
                 new_pred_list_1 = list(total_batch_preds[s1].split(" "))
                 #print("Converted into list")
                 #print(new_pred_list_1) 
                 
                 
                 for j in range(len(new_pred_list_1)):
                     
                     if new_pred_list_1[j] == new_pred_list[-1]:
                         
                         pass
                     
                     else: 
                         
                         new_pred_list.append(new_pred_list_1[j])
                 
                 
                 #print("Cleaned Repeated data")
                 #print(new_pred_list)
                    
                  
                 rem_list = ["$","#","Epsilon"]
                 new_pred_list = [x for x in new_pred_list if x not in rem_list]
                 
               
                 l = ' '.join(new_pred_list)
                 
                 
                 
                 #print("Original output string")
                 #print(total_batch_preds[s1])
                 
                 #print("Converted Output string")
                 #print(l)
                 
                 #print(wer(l,a))
                 #print("Word Error Rate is --------------------------------------------")
                 #print(wer(l,a)*100)
                 word_error_rate[s1] = wer(l,a)*100
                 
                 word_error_rate_list.append(wer(l,a))
            
            train_epoch_loss = train_batch_loss/i
            
            train_loss_list.append(train_epoch_loss)
            
            
            validation_epoch_loss = val_batch_loss/p
            
            val_loss_list.append(validation_epoch_loss)
            
            
            scheduler.step(validation_epoch_loss)
            
            word_error_rate_epoch_list.append(torch.mean(word_error_rate))
            
           
            
                
            print(f'Epoch {n+1} / Epochs {n_epochs} complited with epoch training loss = {train_epoch_loss} and  Validation loss = {validation_epoch_loss} and Validation Word Error Rate = {torch.mean(word_error_rate)}')
            
            
            
            
            print("Validation")
            print("Original Ground Truth :")
            print(a)
            
            print("Predicted Output :")
            print(l)
            
            
            if n%25 == 0:
                
                plt1.plot(train_loss_list, label = "Training Loss");
                plt1.plot(val_loss_list, label = "Validation Loss");
                plt1.xlabel("Epochs");
                plt1.ylabel("Loss");
                plt1.title("Epochs vs Loss");
                plt1.grid();
                plt1.legend();
                plt1.show();
                
                
                
                plt2.plot(word_error_rate_epoch_list, label = "Validation WER")
                plt2.xlabel("Epochs");
                plt2.ylabel("WER");
                plt2.title("Epochs vs WER");
                plt2.grid();
                plt2.legend();
                plt2.show();
                
                
            
                #print(f'output shape after one batch {y_hat.shape}')
        
        #print("pick your input for continuous sign",y_hat.shape)
        
            #y_hat = y_hat.permute(1,0,2)
        
        #print("After permutation",y_hat.size())
            #log_soft_max_values =  torch.nn.functional.log_softmax(y_hat,2)
        
        
                    
        #targets = torch.randint(1, clas,(1,output_sequence_length))
        
        
        #print("Input_Log soft max",log_soft_max_values.size())
        #print("Target_length",targets.size())
    
        #print(f'Index{i}')
        
            #loss = torch.nn.CTCLoss(blank=0)(log_soft_max_values, batch["Glosses"], input_lengths, target_lengths)
        
            #loss.backward()
        
            #optimizer.step()
            
            
                
        #print(f'Epoch {n} / Epochs {n_epochs} complited with loss = {loss.item()}')

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
