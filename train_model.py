# pip3 install torch torchvision torchaudio tqdm matplotlib
# pip3 install torch torchvision torchaudio tqdm matplotlib --index-url https://download.pytorch.org/whl/cu118

import os 
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #（代表仅使用第0，1号GPU）
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from make_train_data import H5HanDataset
from han_comp import HanComp
seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# 设备选择
use_gpu = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")

# 模型参数
batch_size = 200
encoder_embedding_dim = 4
decoder_embedding_dim = 256
encoder_hidden_dim = 512
decoder_hidden_dim = 512
encoder_dropout = 0.5
decoder_dropout = 0.5

class HanDataset(Dataset):
    def __init__(self, filename, han_comp:HanComp, add_han_label=False):
        self.h5db = H5HanDataset(filename)
        self.add_han_label = add_han_label
        self.han_comp = han_comp

    def __getitem__(self, idx):
        item = self.h5db[idx]
        if self.add_han_label:
            han_ord = ord(item['han'])
            point_data = [[0, 0, 0, 0]] + [[han_ord, han_ord, han_ord, han_ord]] + item['points'].tolist() + [[-1, -1, -1, -1]]
        else:
            point_data = [[0, 0, 0, 0]] + item['points'].tolist() + [[-1, -1, -1, -1]]
        point_len = len(point_data)

        labels = item['labels'] 
        label_ids = []
        label_ids.append(self.han_comp.getBosId())
        for label in labels:
            label_ids.append(label)
        label_ids.append(self.han_comp.getEosId())
        point_data = torch.tensor(point_data, dtype=torch.float32)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        
        return label_ids, point_data, point_len

    def __len__(self):
        return len(self.h5db)

class HanCompLabel():

    def __init__(self, han_comp:HanComp):
        self.han_comp = han_comp

    def getBosId(self):
        return 0

    def getEosId(self):
        return self.han_comp.get_comp_total() + 1

    def getLabel(self, idx):
        return self.han_comp.get_comp_name(idx - 1)
    
    def getLabelMaxLen(self):
        return self.han_comp.get_comp_max_num() + 2

    def __getitem__(self, idx):
        return idx
        
    def __len__(self):
        return self.han_comp.get_comp_total() + 2

class HanOrderLabel():

    def __init__(self, han_comp:HanComp):
        self.han_comp = han_comp

    def getBosId(self):
        return 0

    def getEosId(self):
        return self.han_comp.get_stroke_max_num() + 1

    def getLabel(self,  idx):
        return str(idx)
    
    def getLabelMaxLen(self):
        return self.han_comp.get_stroke_max_num() + 2

    def __getitem__(self, idx):
        return idx
        
    def __len__(self):
        return self.han_comp.get_stroke_max_num() + 2

# class HanOrderLabel():

#     def __init__(self, han_comp:HanComp):
#         self.han_comp = han_comp

#     def getBosId(self):
#         return self.han_comp.get_stroke_max_num()

#     def getEosId(self):
#         return self.han_comp.get_stroke_max_num() + 1

#     def getLabel(self,  idx):
#         return self.han_comp.get_comp_name(idx)
    
#     def getLabelMaxLen(self):
#         return self.han_comp.get_stroke_max_num() + 2

#     def __getitem__(self, idx):
#         return idx
        
#     def __len__(self):
#         return self.han_comp.get_stroke_max_num() + 2

class HanStrokeLabel():
    def __init__(self, han_comp:HanComp):
        self.han_comp = han_comp

    def getBosId(self):
        return 0

    def getEosId(self):
        return self.han_comp.get_stroke_total() + 1

    def getLabel(self,  idx):
        return str(idx)

    def getLabelMaxLen(self):
        return self.han_comp.get_stroke_max_num() + 2

    def __getitem__(self, idx):
        return idx
    
    def __len__(self):
        return self.han_comp.get_stroke_total() + 2

class HanStrokeLabel1():
    def __init__(self, han_comp:HanComp):
        self.han_comp = han_comp

    def getBosId(self):
        return self.han_comp.get_stroke_total()

    def getEosId(self):
        return self.han_comp.get_stroke_total() + 1

    def getLabel(self,  idx):
        return str(idx + 1)

    def getLabelMaxLen(self):
        return self.han_comp.get_stroke_max_num() + 2

    def __getitem__(self, idx):
        return idx
    
    def __len__(self):
        return self.han_comp.get_stroke_total() + 2

def collate_fn(batch_data):
    batch_data.sort(key=lambda data: data[2], reverse=True)
    labels = [x[0] for x in batch_data]
    points = [x[1] for x in batch_data]
    points_len = [x[2] for x in batch_data]

    points = nn.utils.rnn.pad_sequence(points, padding_value=0)
    points_len = torch.tensor(points_len, dtype=torch.long)
    batch_labels = nn.utils.rnn.pad_sequence(labels, padding_value=0)

    batch = {
        "points": points,
        "points_len": points_len,
        "labels": batch_labels
    }
    return batch

class Encoder(nn.Module):
    def __init__(self, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional = True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        
        x_packed = nn.utils.rnn.pack_padded_sequence(input=src, lengths=src_len)
        # src = [src length, batch size]
        packed_outputs, hidden = self.rnn(x_packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        # initial decoder hidden is final hidden state of the forwards and backwards 
        # encoder RNNs fed through a linear layer
        
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # outputs = [src length, batch size, encoder hidden dim * 2]
        # hidden = [batch size, decoder hidden dim]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        dropout,
        attention,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((encoder_hidden_dim * 2) + embedding_dim, decoder_hidden_dim) 
        self.fc_out = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim + embedding_dim, 
            output_dim
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, decoder hidden dim]
        # encoder_outputs = [src length, batch size, encoder hidden dim * 2]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, embedding dim]
        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src length]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src length]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, encoder hidden dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, encoder hidden dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        # rnn_input = [1, batch size, (encoder hidden dim * 2) + embedding dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [seq length, batch size, decoder hid dim * n directions]
        # hidden = [n layers * n directions, batch size, decoder hid dim]
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, decoder hidden dim]
        # hidden = [1, batch size, decoder hidden dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        #prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim, 
            decoder_hidden_dim
        )
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, decoder hidden dim]
        # encoder_outputs = [src length, batch size, encoder hidden dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        # repeat decoder hidden state src_length times
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src length, decoder hidden dim]
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2))) 
        # energy = [batch size, src length, decoder hidden dim]
        attention = self.v_fc(energy).squeeze(2)
        # attention = [batch size, src length]
        return torch.softmax(attention, dim=1)
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
        # outputs = [src length, batch size, encoder hidden dim * 2]
        # hidden = [batch size, decoder hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, decoder hidden dim]
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_fn(epoch, model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    sample_total = 0
    train_bar = tqdm.tqdm(data_loader)  # 进度条
    train_bar.set_description(f'[T-{epoch}]')
    train_bar.set_postfix(loss = 0.0)

    for i, data in enumerate(train_bar):
        labels = data['labels'].to(device)
        points = data['points'].to(device)
        points_len = data['points_len'].to('cpu')
        sample_total += points_len.shape[0]

        optimizer.zero_grad()
        output = model(points, points_len, labels, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        labels = labels[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        train_bar.set_postfix(loss = epoch_loss / sample_total)
    return epoch_loss / len(data_loader)

def evaluate_fn(epoch, model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    sample_total = 0
    with torch.no_grad():
        train_bar = tqdm.tqdm(data_loader)
        train_bar.set_description(f'[E-{epoch}]')
        train_bar.set_postfix(loss = 0.0)

        # for i, batch in enumerate(data_loader):
        for batch in train_bar:
            labels = batch['labels'].to(device)
            points = batch['points'].to(device)
            points_len = batch['points_len'].to('cpu')
            sample_total += points_len.shape[0]
            
            output = model(points, points_len, labels, 0) #turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            labels = labels[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            train_bar.set_postfix(loss = epoch_loss / sample_total)
    return epoch_loss / len(data_loader)

def train(han_label, model, model_filename, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, epochs):
    n_epochs = epochs
    clip = 1.0
    teacher_forcing_ratio = 0.5
    best_valid_loss = float("inf")
    begin_epoc = 0

    json_data = {}
    json_data['epoch'] = []

    json_filename = model_filename + '.json'
    if os.path.isfile(json_filename):
        f = open(json_filename)
        json_data = json.load(f)
        f.close()

    begin_epoc = len(json_data['epoch'])
    
    for epoch in range(begin_epoc, n_epochs):
        
        train_loss = train_fn(
            epoch, 
            model, 
            train_data_loader, 
            optimizer, 
            criterion, 
            clip, 
            teacher_forcing_ratio, 
            device,
        )
        
        valid_loss = evaluate_fn(
            epoch, 
            model,
            valid_data_loader, 
            criterion, 
            device,
        )

        filename_ext = model_filename.split(".")[-1]
        epoch_model_filename = "{}.{}.{}".format(model_filename[:-len(filename_ext)-1], epoch, filename_ext)
        torch.save(model.state_dict(), epoch_model_filename)
        
        test_acc = test(han_label, model, epoch_model_filename, test_data_loader, max_num=10000)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_filename)

        epoch_info = {'epoch_id':epoch, 'train_loss': train_loss, 'valid_loss': valid_loss, 'test_acc': test_acc}
        json_data['epoch'].append(epoch_info)

        f = open(json_filename, 'w', encoding='utf-8')
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        f.close()

def plot_attention(sentence, translation, attention):
    fig, ax = plt.subplots(figsize=(10,10))
    attention = attention.squeeze(1).numpy()
    cax = ax.matshow(attention, cmap="bone")
    # ax.set_xticks(ticks=np.arange(len(sentence)), labels=sentence, rotation=90, size=15)
    # translation = translation[1:]
    # ax.set_yticks(ticks=np.arange(len(translation)), labels=translation, size=15)
    plt.show()
    plt.close()


def predict_fn(han_label, batch, model, device, max_output_length=5):
    model.eval()
    
    bos_id = han_label.getBosId()
    eos_id = han_label.getEosId()
    
    with torch.no_grad():
        labels = batch['labels'].to(device)
        points = batch['points'].to(device)
        points_len = batch['points_len'].to("cpu")
        encoder_outputs, hidden = model.encoder(points, points_len)
        inputs = [han_label.getBosId()]
        attentions = torch.zeros(max_output_length, 1, points_len)
        for i in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, attention = model.decoder(inputs_tensor, hidden, encoder_outputs)
            attentions[i] = attention
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == eos_id:
                break

    pred_labels = ''
    for i in inputs:
        if han_label[i] not in [bos_id, eos_id]:
            pred_labels += '{},'.format(han_label.getLabel(han_label[i]))
    if len(pred_labels) > 0:
        pred_labels = pred_labels[:-1]
    # plot_attention('', '', attentions)
    return pred_labels, inputs

def evaluate(model, model_filename, criterion, valid_data_loader):
    model.load_state_dict(torch.load(model_filename))
    test_loss = evaluate_fn(0, model, valid_data_loader, criterion, device)
    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

def test(han_label, model, model_filename, test_data_loader, max_num = -1):
    right = 0
    total = 0
    data_total = len(test_data_loader)
    if (max_num > 0 and max_num < data_total):
        data_total = max_num

    model.load_state_dict(torch.load(model_filename))
    test_iter = iter(test_data_loader)

    train_bar = tqdm.tqdm(range(data_total))  # 进度条
    train_bar.set_description(f'[T-T]')
    train_bar.set_postfix(acc = 0.0)

    for i in train_bar:
        batch = next(test_iter)
        _, predict_label_ids = predict_fn(han_label, batch, model, device, han_label.getLabelMaxLen())
        label_ids = batch['labels']
        label_ids = label_ids.squeeze(dim=1)
        label_ids = label_ids.tolist()
        if (label_ids == predict_label_ids):
            right += 1
        total += 1
        train_bar.set_postfix(acc = right / total)
        if (max_num >0 and total >= max_num):
            break

    return right / total

def predict(han_label, model, model_filename, test_data_loader, show_all = False, max_num=1000):
    model.load_state_dict(torch.load(model_filename))
    right_total = 0

    for i, batch in enumerate(test_data_loader):
        label = ''
        label_id_list = batch['labels'].squeeze().tolist()
        for idx in range(1, len(label_id_list) - 1):
            label += han_label[label_id_list[idx]]
        predict_label, _ = predict_fn(han_label, batch, model, device)
        if (label == predict_label):
            right_total += 1
            if show_all:
                print(i + 1, '  label =', label, 'predict =', predict_label)
        else:
            if show_all:
                print(i + 1, '! label =', label, 'predict =', predict_label)
            else:
                print(i + 1, 'label =', label, 'predict =', predict_label)

        if i + 1 >= max_num:
            break

def predict1(han_label, model, model_filename, test_data_loader, show_all = False, max_num=1000):
    model.load_state_dict(torch.load(model_filename))
    right_total = 0

    for i, batch in enumerate(test_data_loader):
        label = ''
        label_id_list = batch['labels'].squeeze().tolist()
        for idx in range(1, len(label_id_list) - 1):
            label += han_label[label_id_list[idx]]
        predict_label, _ = predict_fn(han_label, batch, model, device)
        if (label == predict_label):
            right_total += 1
            if show_all:
                print(i + 1, '  label =', label, 'predict =', predict_label)
        else:
            if show_all:
                print(i + 1, '! label =', label, 'predict =', predict_label)
            else:
                print(i + 1, 'label =', label, 'predict =', predict_label)

        if i + 1 >= max_num:
            break

def train_model(han_label, name, epochs, add_han_label_dataset=False):
    model_filename =      './output/{}/{}_model.pt'.format(name, name)
    data_train_filename = './data/result/{}_train.h5'.format(name)
    data_val_filename =   './data/result/{}_val.h5'.format(name)
    data_test_filename =  './data/result/{}_test.h5'.format(name)

    output_dim = len(han_label)
    attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

    encoder = Encoder(
        encoder_embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        encoder_dropout,
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        decoder_dropout,
        attention,
    )

    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    model = Seq2Seq(encoder, decoder, device).to(device)      
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_data_loader = DataLoader(HanDataset(data_train_filename, han_label, add_han_label_dataset), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_data_loader = DataLoader(HanDataset(data_val_filename, han_label, add_han_label_dataset), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_data_loader = DataLoader(HanDataset(data_test_filename, han_label, add_han_label_dataset), batch_size=1, collate_fn=collate_fn, shuffle=True)
    # oov_data_loader = DataLoader(HanDataset(data_oov_filename, han_label), batch_size=1, collate_fn=collate_fn, shuffle=False)

    print(f"The model has {count_parameters(model):,} trainable parameters")
    train(han_label, model, model_filename, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, epochs=epochs)
    # evaluate(model, model_filename, criterion, valid_data_loader)
    # test(han_label, model, model_filename, test_data_loader, max_num=10000)
    # predict(han_label, model, model_filename, oov_data_loader, show_all=True, max_num=100)

def predict_order(han_label, model_filename, test_filename, device):
    output_dim = len(han_label)
    attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

    encoder = Encoder(
        encoder_embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        encoder_dropout,
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        decoder_dropout,
        attention,
    )

    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    model = Seq2Seq(encoder, decoder, device).to(device)
    if device == 'cpu':
        model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_filename))

    batch = {}
    f = open(test_filename, mode='r', encoding='utf_8')
    for each in f:
        batch = json.loads(each)
        point_data = []
        label_ids = []
        point_data.append([[0, 0, 0, 0]])
        for item in batch['points']:
            point_data.append([item])
        point_data.append([[-1, -1, -1, -1]])

        point_len = len(point_data)

        labels = batch['labels'] 
        label_ids.append([han_label.getBosId()])
        for label in labels:
            label_ids.append([label])
        label_ids.append([han_label.getEosId()])
        point_data = torch.tensor(point_data, dtype=torch.float32)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        
        points = point_data
        points_len = torch.tensor([point_len], dtype=torch.long)
        batch_labels = label_ids

        batch = {
            "points": points,
            "points_len": points_len,
            "labels": batch_labels
        }

        predict_label, _ = predict_fn(han_label, batch, model, device, len(han_label))
        print(predict_label)

class HanOrderModel:
    def __init__(self, han_filename, comp_filename, model_filename, device = 'cpu'):
        self.device = device
        self.han_comp = HanComp(han_filename, comp_filename)
        self.han_order_label = HanOrderLabel(self.han_comp)
        output_dim = len(self.han_order_label)
        attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

        encoder = Encoder(
            encoder_embedding_dim,
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_dropout,
        )

        decoder = Decoder(
            output_dim,
            decoder_embedding_dim,
            encoder_hidden_dim,
            decoder_hidden_dim,
            decoder_dropout,
            attention,
        )

        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(model_filename))

    def predict_fn(self, han_label, batch, model, device, max_output_length=5):
        self.model.eval()
        
        bos_id = han_label.getBosId()
        eos_id = han_label.getEosId()
        
        with torch.no_grad():
            points = batch['points'].to(device)
            points_len = batch['points_len'].to("cpu")
            encoder_outputs, hidden = model.encoder(points, points_len)
            inputs = [han_label.getBosId()]
            attentions = torch.zeros(max_output_length, 1, points_len)
            for i in range(max_output_length):
                inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
                output, hidden, attention = model.decoder(inputs_tensor, hidden, encoder_outputs)
                attentions[i] = attention
                predicted_token = output.argmax(-1).item()
                inputs.append(predicted_token)
                if predicted_token == eos_id:
                    break

        pred_labels = ''
        for i in inputs:
            if han_label[i] not in [bos_id, eos_id]:
                pred_labels += '{},'.format(han_label.getLabel(han_label[i]))
        if len(pred_labels) > 0:
            pred_labels = pred_labels[:-1]
        # plot_attention('', '', attentions)
        return pred_labels, inputs

    def get_order(self, strokes):

        point_data = []
        label_ids = []
        point_data.append([[0, 0, 0, 0]])

                # calc rect
        l = sys.maxsize
        t = sys.maxsize
        r = 0
        b = 0
        for stroke in strokes:
            for point in stroke:
                x = point['x']
                y = point['y']

                l = x if x < l else l
                t = y if y < t else t
                r = x if x > r else r
                b = y if y > b else b

        max_len = max(r - l, b - t)
        
        # for stroke in strokes:
        #     for point in stroke:
        #         point['x'] -= l
        #         point['y'] -= t
        

        for i, stroke in enumerate(strokes):
            for j, points in enumerate(stroke):
                x = points['x']
                y = points['y']
                x = int((x - l) * 256 / max_len)
                y = int((y - t) * 256 / max_len)
                point_data.append([[i, j, x, y]])

        
        point_data.append([[-1, -1, -1, -1]])
        print(point_data)
        point_len = len(point_data)

        point_data = torch.tensor(point_data, dtype=torch.float32)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        
        points = point_data
        points_len = torch.tensor([point_len], dtype=torch.long)
        batch_labels = label_ids

        batch = {
            "points": points,
            "points_len": points_len
        }

        predict_label, _ = self.predict_fn(self.han_order_label, batch, self.model, device, len(self.han_order_label))
        print(predict_label)
        return predict_label

if __name__ == "__main__":

    han_filename = './labels/han.jsonl'
    comp_filename = './labels/comp.jsonl'

    han_comp = HanComp(han_filename, comp_filename)
    han_comp_label = HanCompLabel(han_comp)
    han_order_label = HanOrderLabel(han_comp)
    han_stroke_label = HanStrokeLabel(han_comp)

    model_filename = './output/{}/{}_model.4.pt'.format('han_sorder_palm_4f60', 'han_sorder_palm_4f60')
    
    predict_order(han_order_label, model_filename, './data/result/han_sorder_palm_4f60_test.jsonl', 'cpu')
    # train_model(han_comp_label, 'han_comp_extr_casia', epochs = 100, add_han_label_dataset=False)
    # train_model(han_order_label, 'han_sorder_palm_6728', epochs = 100, add_han_label_dataset=False)
    # train_model(han_order_label, 'han_sorder_palm_6c49', epochs = 100, add_han_label_dataset=False)
    # train_model(han_comp_label, 'han_comp_extr_palm_920', epochs = 100, add_han_label_dataset=False)
    # train_model(han_stroke_label, 'han_stroke_palm_920', epochs = 100, add_han_label_dataset=False)
    train_model(han_order_label, 'han_sorder_palm_4f60', epochs = 100, add_han_label_dataset=False)
    
    # train_model(han_order_label, 'han_sorder_scut', epochs = 100, add_han_label_dataset=True)
