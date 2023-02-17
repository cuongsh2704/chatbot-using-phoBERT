
import torch
import torchcrf
import numpy as np
from torch import nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torchcrf import CRF
from pre_train_phobert import pre_train

tokenizer, phoBERT = pre_train()

#IC

label = ['greetings',
 'confirm',
 'thank_you',
 'confirm#thank_you',
 'inter_time',
 'want_apply',
 'inter_time_type',
 'not_suitable',
 'inter_position',
 'require_jd',
 'require_jd#inter_position',
 'job_requirement',
 'how_to_apply',
 'confirm_email',
 'confirm_submit_CV',
 'personal_experience',
 'salary',
 'response',
 'if_recruit',
 'training_plan',
 'position_slot',
 'apply_deadline']
LABEL_SIZE = len(label)
class CBmodel(nn.Module):
  def __init__(self, pre_train, n_dim, num_label):
    super().__init__()
    #implement pre-train model phoBERT
    self.BERT = pre_train
    #freeze BERT layer
    for child in self.BERT.children():
      for param in child.parameters():
          param.requires_grad = False
    #fully connected
    self.fc_ic = nn.Linear(n_dim, num_label)
    #softmax layer
    self.softmax = nn.Softmax(dim=1)
  def forward(self, x, tags=None, pad_id=1.0):
    #masking
    x_masks = self.make_bert_mask(x, pad_id)  # (B, L)
    features = self.BERT(x, attention_mask=x_masks) #(B, L, N_dim) , (B, N_dim)
    output_ic = self.fc_ic(features[1]) #(B, N_labels)
    output_ic = self.softmax(output_ic)
    return output_ic

  def make_bert_mask(self, x, pad_id):
    bert_masks = (x != pad_id).float()  # (B, L)
    return torch.Tensor(bert_masks)


#test phase

def vectorlization(sent):
  seq = tokenizer.encode(sent)
  seq_pad = pad_sequences([seq], 50, padding='post', value=1.0)
  return torch.tensor(seq_pad)

def predict(sent, model):
    with torch.no_grad():
      input = vectorlization(sent)
      output = model(input)
    ids = torch.argmax(output[0])
    return label[ids]
if __name__ == "__main__":

    model_ic = CBmodel(phoBERT, 768, LABEL_SIZE)
    model_ic = torch.load("D:\Flask fundamental\model\CBmodel_Ver1.pth", map_location=torch.device('cpu'))
    model_ic.eval()
    print(predict("chỗ mình còn tuyển tts không "))