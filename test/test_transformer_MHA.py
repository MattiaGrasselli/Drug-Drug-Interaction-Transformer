import os
import sys
sys.path.append(f'..{os.sep}')

import torch
from dataset.dataset import *
from model.utils.config import DrugConfig
from model.DDItransformer import TransformerDDI_MHA
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score,f1_score,accuracy_score


parser=argparse.ArgumentParser(description="Drug-Drug Interaction test for TransformerDDI with Multi-head self-attention fusion")
parser.add_argument("--test-csv", type=str, dest="test_csv", help="Path to the test csv")
parser.add_argument("--output-dir", type=str, dest="output_dir", help="Path to the output where test file will be placed")
parser.add_argument("--vocabulary", type=str, dest="vocab", help="Path to the vocabulary file")
parser.add_argument("--model-weights", type=str, dest="weights", help="Path to the trained model weights")
parser.add_argument("--num-encoders", type=int, dest="encoders", default=8, help="Number of encoders")
parser.add_argument("--dropout", type=float, dest="dropout", default=0.1, help="Dropout probability")
parser.add_argument("--multi-head", type=int, dest="multi_head", default=8, help="Number of heads in the multi-head self-attention utilized as feature fusion")
parser.add_argument("--batch-size", dest="BATCH_SIZE",type=int, default=256, help="Testing Batch-size")
parser.add_argument("--num-workers",dest="NUM_WORKERS",type=int, default=1, help="Number of workers")

args=parser.parse_args()

BATCH_SIZE=args.BATCH_SIZE
NUM_WORKERS=args.NUM_WORKERS

#Here i specify the file_csv name for test
test_csv=args.test_csv

#Output directory where the test_metrics_DDI.csv will be saved
output_dir=args.output_dir

#Model weights 
model_weights=args.weights

#Model number of multi-head
num_multi_head=args.multi_head

#DrugConfig specifications
num_encoders=args.encoders
dropout=args.dropout

#Vocabulary
vocabulary=args.vocab

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

left_config=DrugConfig(num_hidden_layers=num_encoders,dropout_prob=dropout)
right_config=DrugConfig(num_hidden_layers=num_encoders,dropout_prob=dropout)

model=TransformerDDI_MHA(left_config,right_config,num_multi_head=num_multi_head)
model.to(device)
model.load_state_dict(torch.load(model_weights,map_location=device),strict=True)

#Here Datasets are defined
test_dataset=DDIdataset(test_csv,vocabulary,left_config)

#Here there are the DataLoaders
test_loader=DataLoader(test_dataset,BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

if __name__=='__main__':
    metrics=[]
        
    labels=torch.Tensor()
    predictions=torch.Tensor()

    #-->Testing is performed<--
    model.eval()

    print("Start processing")

    for bathc_id, (drug1_ID,drug1_token_id,drug1_mask,drug2_ID,drug2_token_id,drug2_mask, label) in enumerate(test_loader):
        #print(f"Sono l\'input: {drug1_ID},{drug1_token_id},{drug1_mask},{drug2_ID},{drug2_token_id},{drug2_mask}")
        #print(f"Sono la label: {label}")

        with torch.no_grad():
            drug1_token_id,drug1_mask,drug2_token_id,drug2_mask, label=\
                drug1_token_id.to(device),drug1_mask.to(device),\
                drug2_token_id.to(device),drug2_mask.to(device), label.to(device)

            #Forward-pass
            label_pred=model(drug1_token_id,drug1_mask,drug2_token_id,drug2_mask)
            
            #I need to decrement label by 1 because in the file they go from 1 to 86
            #but the loss needs values from 0 to 85
            label-=1

            _,pred=torch.max(label_pred.data,1)

            predictions=torch.cat([predictions,pred.cpu()])
            labels=torch.cat([labels,label.cpu()])
            #print(predictions.shape)

    print("End Processing")

    micro_precision_test=precision_score(labels,predictions,average="micro")
    macro_precision_test=precision_score(labels,predictions,average="macro")
    micro_recall_test=recall_score(labels,predictions,average="micro")
    macro_recall_test=recall_score(labels,predictions,average="macro")
    micro_f1_test=f1_score(labels,predictions,average="micro")
    macro_f1_test=f1_score(labels,predictions,average="macro")
    accuracy_test=accuracy_score(labels,predictions)
        
    metrics.append([accuracy_test,micro_precision_test,macro_precision_test,\
                    micro_recall_test,macro_recall_test,micro_f1_test,macro_f1_test])
    
    print(f"accuracy: {metrics[0][0]}, micro-average precision: {metrics[0][1]}, macro-average precision: {metrics[0][2]}, micro-average recall: {metrics[0][3]}, macro-average recall: {metrics[0][4]}, micro-average F1-score: {metrics[0][5]}, macro-average F1-score: {metrics[0][6]}")
    
    #print(metrics)
    #I save the metrics in a csv file
    metrics_file_csv=pd.DataFrame(metrics,columns=['accuracy_test','micro_precision_test','macro_precision_test',\
                    'micro_recall_test','macro_recall_test','micro_f1_test','macro_f1_test'])
    metrics_file_csv.to_csv(f'{output_dir}{os.sep}test_metrics_DDI.csv',sep=";")
