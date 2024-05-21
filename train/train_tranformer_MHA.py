import sys
import os
sys.path.append(f'..{os.sep}')

import torch
from dataset.dataset import *
from model.utils.config import DrugConfig
from model.DDItransformer import TransformerDDI_MHA
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import get_scheduler
from sklearn.metrics import recall_score, precision_score,f1_score,accuracy_score


parser=argparse.ArgumentParser(description="Drug-Drug Interaction train")
parser.add_argument("--training-csv", dest="training_csv", type=str,help="Path to the training csv")
parser.add_argument("--validation-csv", dest="validation_csv", type=str,help="Path to the validation csv")
parser.add_argument("--output-dir", type=str, dest="output_dir", help="Path to the output where results will be placed")
parser.add_argument("--vocabulary",dest="vocabulary",type=str,help="Path to the vocabulary")
parser.add_argument("--num-encoders", type=int, dest="encoders", default=8, help="Number of encoders")
parser.add_argument("--dropout", type=float, dest="dropout", default=0.1, help="Dropout probability")
parser.add_argument("--multi-head", type=int, dest="multi_head", default=8, help="Number of heads in the multi-head self-attention utilized as feature fusion")
parser.add_argument("--epochs", dest="N_EPOCH", type=int, default=200, help="Total number of epochs")
parser.add_argument("--batch-size", dest="BATCH_SIZE",type=int, default=256, help="Batch size dimensionality.")
parser.add_argument("--learning-rate", dest="LEARNING_RATE",type=float, default=1e-4, help="Learning rate")
parser.add_argument("--betas",dest="BETAS",type=float,nargs='+',default=(0.9,0.999),help="Beta values used in AdamW optimizer. For instance, --betas 0.9 0.999")
parser.add_argument("--eps",dest="EPS",type=float,default=1e-6,help="Epsilon value used in AdamW optimizer")
parser.add_argument("--weight-decay", dest="WEIGHT_DECAY",type=float, default=0.01, help="Weight-Decay")
parser.add_argument("--num-workers",dest="NUM_WORKERS",type=int, default=1, help="Number of workers")
parser.add_argument("--early-stopping",dest="EARLY_STOPPING",type=int,default=10,help="After early_stopping \
                    epochs without better loss, Early Stopping happens")

args=parser.parse_args()

N_EPOCH=args.N_EPOCH
BATCH_SIZE=args.BATCH_SIZE
LEARNING_RATE=args.LEARNING_RATE
BETAS=tuple(args.BETAS)
EPS=args.EPS
WEIGHT_DECAY=args.WEIGHT_DECAY
NUM_WORKERS=args.NUM_WORKERS
EARLY_STOPPING=args.EARLY_STOPPING

#Here it is specified the output directory
output_dir=args.output_dir

#Here i specify the file_csv names for training, validation and test
training_csv=args.training_csv
validation_csv=args.validation_csv

#Here it is specified the vocabulary
vocabulary=args.vocabulary

#Here they are specified the model characteristics
num_multi_head=args.multi_head
num_encoders=args.encoders
dropout=args.dropout

left_config=DrugConfig(num_hidden_layers=num_encoders,dropout_prob=dropout)
right_config=DrugConfig(num_hidden_layers=num_encoders,dropout_prob=dropout)
model=TransformerDDI_MHA(left_config,right_config,num_multi_head=num_multi_head)

#Here Datasets are defined
training_dataset=DDIdataset(training_csv,vocabulary,config=left_config)
validation_dataset=DDIdataset(validation_csv,vocabulary,config=left_config)

#Here there are the DataLoaders
training_loader=DataLoader(training_dataset,BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
validation_loader=DataLoader(validation_dataset,BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

#Here the loss function is definined
loss_function=nn.CrossEntropyLoss()

model_parameters=model.parameters()
#for i in model.named_parameters():
#    print(i)

num_training_steps = N_EPOCH * len(training_loader)

opt=torch.optim.AdamW(model_parameters,lr=LEARNING_RATE,betas=BETAS,eps=EPS,weight_decay=WEIGHT_DECAY)
lr_scheduler=get_scheduler(name="linear",optimizer=opt,num_warmup_steps=num_training_steps/N_EPOCH,num_training_steps=num_training_steps)

if __name__=='__main__':
    min_loss=np.Inf
    is_early_stopping_happened=False

    metrics=[]

    #Training and validation is performed
    for epoch in range(0, N_EPOCH):
        print(f"This is the epoch number: {epoch+1}")

        labels=torch.Tensor()
        predictions=torch.Tensor()

        mean_train_loss=0.0
        mean_train_accuracy=0.0

        mean_validation_loss=0.0
        mean_validation_accuracy=0.0

        #-->Train is performed<--
        model.train()

        for bathc_id, (drug1_ID,drug1_token_id,drug1_mask,drug2_ID,drug2_token_id,drug2_mask, label) in enumerate(training_loader):
            #print(f"Sono l\'input: {drug1_ID},{drug1_token_id},{drug1_mask},{drug2_ID},{drug2_token_id},{drug2_mask}")
            #print(f"Sono la label: {label}")
            
            drug1_token_id,drug1_mask,drug2_token_id,drug2_mask, label=\
                    drug1_token_id.to(device),drug1_mask.to(device),\
                    drug2_token_id.to(device),drug2_mask.to(device), label.to(device)

            #Set the gradient to 0
            opt.zero_grad()

            #Forward-pass
            label_pred=model(drug1_token_id,drug1_mask,drug2_token_id,drug2_mask)
            
            #I need to decrement label by 1 because in the file they go from 1 to 86
            #but the loss needs values from 0 to 85
            label-=1
            #Calculate the loss
            loss=loss_function(label_pred,label)
            
            #Backward-pass
            loss.backward()

            #Update parameters
            opt.step()
            lr_scheduler.step()

            #Track train loss by multiplying the average loss by the number of examples in the batch
            mean_train_loss += loss.item()*BATCH_SIZE 

            _,pred=torch.max(label_pred.data,1)

            predictions=torch.cat([predictions,pred.cpu()])
            labels=torch.cat([labels,label.cpu()])
            #print(predictions.shape)

        # Compute average loss
        mean_train_loss=mean_train_loss/len(training_loader.dataset)

        micro_precision_train=precision_score(labels,predictions,average="micro")
        macro_precision_train=precision_score(labels,predictions,average="macro")
        micro_recall_train=recall_score(labels,predictions,average="micro")
        macro_recall_train=recall_score(labels,predictions,average="macro")
        micro_f1_train=f1_score(labels,predictions,average="micro")
        macro_f1_train=f1_score(labels,predictions,average="macro")
        accuracy_train=accuracy_score(labels,predictions)
        
        #print(f"Valore accuracy: {accuracy_train}, valore micro precision: {micro_precision_train}, valore macro precision: {macro_precision_train}\
        #      ,valore di macro recall: {macro_recall_train}, valore di micro recall: {micro_recall_train}")

        labels=torch.Tensor()
        predictions=torch.Tensor()

        #-->Validation is performed<--
        model.eval()

        for drug1_ID,drug1_token_id,drug1_mask,drug2_ID,drug2_token_id,drug2_mask, label in validation_loader:
            #print(f"This is the value of inputs: {inputs.shape}")
            #print(f"This is the value of labels: {labels}")

            with torch.no_grad():
                drug1_token_id,drug1_mask,drug2_token_id,drug2_mask, label=\
                        drug1_token_id.to(device),drug1_mask.to(device),\
                        drug2_token_id.to(device),drug2_mask.to(device), label.to(device)
    
                #Forward Pass
                label_pred=model(drug1_token_id,drug1_mask,drug2_token_id,drug2_mask)
    
                label-=1
                #Validation Loss
                loss=loss_function(label_pred,label)
                #I multiply the loss with the number of examples in the batch
                mean_validation_loss+=loss.item()*BATCH_SIZE
    
                #Compute validation accuracy
                _,pred=torch.max(label_pred.data,1)
    
                predictions=torch.cat([predictions,pred.cpu()])
                labels=torch.cat([labels,label.cpu()])
                #print(predictions.shape)

        #Average Loss
        mean_validation_loss=mean_validation_loss/len(validation_loader.dataset)

        #print(f"This is the mean loss: {mean_validation_loss}")
        
        micro_precision_val=precision_score(labels,predictions,average="micro")
        macro_precision_val=precision_score(labels,predictions,average="macro")
        micro_recall_val=recall_score(labels,predictions,average="micro")
        macro_recall_val=recall_score(labels,predictions,average="macro")
        micro_f1_val=f1_score(labels,predictions,average="micro")
        macro_f1_val=f1_score(labels,predictions,average="macro")
        accuracy_val=accuracy_score(labels,predictions)

        #print(f"Valore accuracy: {accuracy_val}, valore micro precision: {micro_precision_val}, valore macro precision: {macro_precision_val}\
        #      ,valore di macro recall: {macro_recall_val}, valore di micro recall: {micro_recall_val}")

        metrics.append([accuracy_train,micro_precision_train,macro_precision_train,\
                        micro_recall_train,macro_recall_train,micro_f1_train,macro_f1_train,\
                        mean_train_loss,accuracy_val,micro_precision_val,macro_precision_val,\
                        micro_recall_val,macro_recall_val,micro_f1_val,macro_f1_val,mean_validation_loss])
        print(metrics)
        #I save the metrics in a csv file
        metrics_file_csv=pd.DataFrame(metrics,columns=['accuracy_train','micro_precision_train','macro_precision_train',\
                        'micro_recall_train','macro_recall_train','micro_f1_train','macro_f1_train',\
                        'mean_train_loss','accuracy_val','micro_precision_val','macro_precision_val',\
                        'micro_recall_val','macro_recall_val','micro_f1_val','macro_f1_val','mean_validation_loss'])
        metrics_file_csv.to_csv(f'{output_dir}{os.sep}train_validation_metrics_DDI.csv',sep=";")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict':lr_scheduler.state_dict(),
            'loss': mean_validation_loss
        },os.path.join(f'{output_dir}',"last_DDI_checkpoint.pth.tar"))
        
        if mean_validation_loss <  min_loss:
            torch.save(model.state_dict(),os.path.join(f'{output_dir}',"DDI_best.pt"))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict':lr_scheduler.state_dict(),
                'loss': mean_validation_loss
            },os.path.join(f'{output_dir}',"DDI_checkpoint.pth.tar"))
            
            #print(metrics)

            min_loss=mean_validation_loss
            best_epoch=epoch
            no_improvement=0
        #The following will represent an Early Stopping
        else:
            no_improvement+=1
            if no_improvement >= EARLY_STOPPING:
                print(f"Early stopping has happened. The best epoch: {best_epoch} with loss: {min_loss}.\nTotal number of epoch performed before Early Stopping: {epoch}")
                is_early_stopping_happened=True
                break
