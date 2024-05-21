# Drug-Drug Interaction Transformer (Transformer-DDI)
This repository contains the AI-for-Bioinformatics exam project.

## Abstract
<p style="text-align: justify;">
<em>Drug-drug interaction (DDI) is a phenomenon that occurs when two (or more) drugs produce unexpected results that might cause harm to a person. As of today, many approaches have been proposed to solve this problem, mainly based on GNN-based models or similarity-based models. In this paper, by considering SMILES strings as a sequence of words, it will be described an architecture based on transformers, namely TransformerDDI, that aims at classifying the DDIs present between two drugs. The transformer architecture first extracts the drug features and then by exploiting a feature fusion and classification it will be able to define which DDI is present.
After performing multiple tests on real-world data, it can be concluded that the approach utilized works, reaching a macro-average F1-score of 92.0%, almost 5% higher with respect to the best non-multimodal reference. Thus, this proves that by analysing SMILES strings as sequence of words, natural language processing approaches could be utilized for performing DDI task too and thus boosting the overall performances.

</em>
</p>

## Folder analysis
• `/dataset`: contains the dataset employed during the training, validation and test phases, the code utilized for performing the split and the dataset classes exploited for the training and testing phases.   
• `/tokenization`: contains the script regarding the tokenization process.  
• `/model`: contains the Transformer-DDI models.  
• `/vocab`: contains the vocabulary utilized for training and testing.  
• `/train`: contains the training scripts for all the models reported in /model.  
• `/test`: contains the testing scripts for all the models reported in /model.

## Environment preparation
1) Download this GitHub repository
2) Create a conda environment by performing: conda create -n &lt;name&gt; python=3.9.18
3) Activate the conda environment by performing: conda activate &lt;name&gt;
4) Install the requirements by performing: "pip install -r requirements.txt"
5) It is now possible to test the model you prefer. See 'How to perform tests' section for further details. 

Note: &lt;name&gt; should be substitued with a name of your choice.

## Model weights
Model weights have been kept private.  

## How to perform tests
1) Go to the `/test` folder. (cd /test)
2) Decide which model you want to test. You can test:  
• Transformer_DDI_FCL (test_transformer_FCL.py): model version which uses the fully-connected layers as feature fusion and classification. (This version is the one which has proved to be the most effective one).  
• Transformer_DDI_MHA (test_transformer_MHA.py): model version which uses multi-head self-attention as feature fusion.  
• Transformer_DDI_Sequence (test_transformer_Sequence.py): version which processes SMILES strings as sentences.  
3) See which are the possible parameters that can be passed by exploiting  --help. For instance, if we want to test the Transformer_DDI_FCL model, perform: python test_transformer_FCL.py --help 

IMPORTANT NOTE 1: all the variables have been already set given the model weights provided. Thus, if you want to test those configurations, the only parameters that need to be set are:  
• --test-csv  
• --output-dir  
• --vocabulary  
• --model-weights  

IMPORTANT NOTE 2: vocab_smiles.txt in `/vocab` should be used for Transformer_DDI_FCL and Transformer_DDI_MHA models. vocab_smiles_sequence.txt should be used for the Transformer_DDI_Sequence model.

## How to train the models
1) Go to the `/train` folder. (cd /train)
2) Decide which model you want to train. You can train:  
• Transformer_DDI_FCL (train_transformer_FCL.py)   
• Transformer_DDI_MHA (train_transformer_MHA.py)  
• Transformer_DDI_Sequence (train_transformer_Sequence.py)  
3) See which are the possible parameters that can be passed by exploiting  --help. For instance, if we want to train the Transformer_DDI_FCL model, perform: python train_transformer_FCL.py --help 

## References
[1] R. P. Riechelmann & A. Del Giglio. “Drug interactions in oncology: how common are they?”  
[2] Dima M. Qato, Jocelyn Wilder, L. Philip Schumm, Victoria Gillet, G. Caleb Alexander. “Changes in Prescription and Over-the-Counter Medication and Dietary Supplement Use Among Older Adults in the United States, 2005 vs 2011”  
[3] Igho J. Onakpoya, Carl J. Heneghan and Jeffrey K. Aronson. “Post-marketing withdrawal of 462 medicinal products because of adverse drug reactions: a systematic review of the world literature.”  
[4] David S. Wishart, Yannick D. Feunang, An C. Guo, Elvis J. Lo, Ana Marcu, Jason R. Grant, Tanvir Sajed, Daniel Johnson, Carin Li, Zinat Sayeeda, Nazanin Assempour, Ithayavani Iynkkaran, Yifeng Liu, Adam Maciejewski, Nicola Gale, Alex Wilson, Lucy Chin, Ryan Cummings, Diana Le, Allison Pon, Craig Knox and Michael Wilson. “DrugBank 5.0: a major update to the DrugBank database for 2018”.  
[5] Jae Yong Ryu, Hyun Uk Kim, and Sang Yup Lee. “Deep learning improves prediction of drug–drug and drug–food interactions”.  
[6] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. “Attention is All You Need”.  
[7] Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”.  
[8] Arnold K. Nyamabo, Hui Yu and Jian-Yu Shi. “SSI–DDI: substructure–substructure interactions for drug–drug interaction prediction”  
[9] Tengfei Lyu, Jianliang Gao, Ling Tian, Zhao Li, Peng Zhang and Ji Zhang. “MDNN: A Multimodal Deep Neural Network for Predicting Drug-Drug Interaction Events”.  
[10] Yifan Deng, Xinran Xu, Yang Qiu, Jingbo Xia, Wen Zhang and Shichao Liu. “A multimodal deep learning framework for predicting drug–drug interaction events”.
