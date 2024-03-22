import json
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from transformers import BertModel, BertTokenizerFast, AdamW
from torch.utils.data import Dataset, DataLoader
from evaluate import load
import matplotlib.pyplot as plt

wer = load("wer")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_data(path): 
    contexts = []
    questions = []
    answers = []

    with open(path, 'rb') as f:
        raw_data = json.load(f)

    for group in raw_data['data']:
        for paragraph in group['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context.lower())
                    questions.append(question.lower())
                    answers.append(answer)
    return contexts, questions, answers

def add_answer_end(answers, contexts):
    for answer, context in zip(answers, contexts):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

def process_long_para(train_contexts, train_answers, max_length):
    train_contexts_pro = []

    for i, context in enumerate(train_contexts):
        if len(context) > max_length:
            answer_start = train_answers[i]['answer_start']
            answer_end = answer_start + len(train_answers[i]['text'])
            mid_point = (answer_start + answer_end) // 2
            start_point = max(0, min(mid_point - max_length // 2, len(context) - max_length))
            end_point = start_point + max_length

            train_contexts_pro.append(context[start_point:end_point])
            train_answers[i]['answer_start'] = answer_start - start_point
        else:
            train_contexts_pro.append(context)

    return train_contexts_pro

train_path = 'spoken_train-v1.1.json'
test_path = 'spoken_test-v1.1.json'
train_contexts, train_questions, train_answers = load_data(train_path)
test_contexts, test_questions, test_answers = load_data(test_path)

add_answer_end(train_answers, train_contexts)
add_answer_end(test_answers, test_contexts)

train_contexts_pro = process_long_para(train_contexts, train_answers, 512)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_questions, train_contexts_pro, 
                            max_length = 512, 
                            truncation=True,
                            padding=True)
test_encodings = tokenizer(test_questions, test_contexts, 
                            max_length = 512, 
                            truncation=True,
                            padding=True)

def find_answer_position(idx, encodings):
    start_position = 0
    end_position = 0
    answer_encoding = tokenizer(train_answers[idx]['text'],  max_length = 512, truncation=True, padding=True)
    for a in range( len(encodings['input_ids'][idx]) -  len(answer_encoding['input_ids']) ):
        match = True
        for i in range(1,len(answer_encoding['input_ids']) - 1):
            if (answer_encoding['input_ids'][i] != encodings['input_ids'][idx][a + i]):
                match = False
                break
            if match:
                start_position = a+1
                end_position = a+i+1
                break
    return(start_position, end_position)

train_start_positions = []
train_end_positions = []
test_start_positions = []
test_end_positions = []
for idx in range(len(train_encodings['input_ids'])):
    start, end = find_answer_position(idx, train_encodings)
    train_start_positions.append(start)
    train_end_positions.append(end)
for idx in range(len(test_encodings['input_ids'])):
    start, end = find_answer_position(idx, test_encodings)
    test_start_positions.append(start)
    test_end_positions.append(end)

train_encodings.update({'start_positions': train_start_positions, 'end_positions': train_end_positions})
test_encodings.update({'start_positions': test_start_positions, 'end_positions': test_end_positions})

class InputDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, i):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][i]),
            'token_type_ids': torch.tensor(self.encodings['token_type_ids'][i]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][i]),
            'start_positions': torch.tensor(self.encodings['start_positions'][i]),
            'end_positions': torch.tensor(self.encodings['end_positions'][i])
        }
    def __len__(self):
        return len(self.encodings['input_ids'])
    
train_dataset = InputDataset(train_encodings)
test_dataset = InputDataset(test_encodings)

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1)

bert_model = BertModel.from_pretrained('bert-base-uncased')

class SimplifiedQAModel(nn.Module):
    def __init__(self):
        super(SimplifiedQAModel, self).__init__()
        self.bert = bert_model 
        self.drop_out = nn.Dropout(0.1)
        self.l1 = nn.Linear(768, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        model_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = model_output[2]
        out = hidden_states[-1]
        out = self.drop_out(out)
        logits = self.l1(out)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

model = SimplifiedQAModel()

def cross_entropy_loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    loss_start = ce_loss(start_logits, start_positions)
    loss_end = ce_loss(end_logits, end_positions)
    return (loss_start + loss_end) / 2

optim = AdamW(model.parameters(), lr=1e-5)

def train_model(model, dataloader):
    model = model.train()
    losses = []
    acc = []
    for batch in tqdm(dataloader, desc = 'Training Model'):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        out_start, out_end = model(input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

        loss = cross_entropy_loss_fn(out_start, out_end, start_positions, end_positions)
        losses.append(loss.item())
        loss.backward()
        optim.step()
        
        start_pred = torch.argmax(out_start, dim=1)
        end_pred = torch.argmax(out_end, dim=1)
            
        acc.append(((start_pred == start_positions).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_positions).sum()/len(end_pred)).item())

    avg_accuracy = sum(acc)/len(acc)
    avg_loss = sum(losses)/len(losses)
    return avg_accuracy, avg_loss

def evaluate_model(model, dataloader):
    model = model.eval()
    answer_list=[]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc = 'Evaluating Model'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            
            out_start, out_end = model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)

            start_pred = torch.argmax(out_start)
            end_pred = torch.argmax(out_end)
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_pred:end_pred]))
            tanswer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_true[0]:end_true[0]]))
            answer_list.append([answer,tanswer])

    return answer_list

model.to(device)
train_accuracies=[]
train_losses=[]
wer_scores=[]

for epoch in range(20):
    train_accuracy, train_loss = train_model(model, train_data_loader)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)
    answer_list = evaluate_model(model, test_data_loader)
    pred_answers=[]
    true_answers=[]
    for i in range(len(answer_list)):
        if(len(answer_list[i][0])==0):
            answer_list[i][0]="$"
        if(len(answer_list[i][1])==0):
            answer_list[i][1]="$"
        pred_answers.append(answer_list[i][0])
        true_answers.append(answer_list[i][1])
    wer_score = wer.compute(predictions=pred_answers, references=true_answers)
    wer_scores.append(wer_score)
    print(f'Epoch: {epoch+1} | Training Accuracy: {train_accuracy:.2f} | Training Loss: {train_loss:.2f} | WER Score: {wer_score:.2f}')

# Save the model
torch.save(model, "trained_model.h5")
print(train_accuracies, train_losses, wer_scores)

epochs = range(1, 21)
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(epochs, train_accuracies, color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
ax2.plot(epochs, train_losses, color=color, label='Loss')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Training Accuracy and Loss')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.show()


plt.figure(figsize=(10, 5))
plt.plot(epochs, wer_scores, color='tab:green', marker='o', linestyle='-', label='WER Score')
plt.title('WER Scores Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('WER Score')
plt.xticks(ticks=epochs)  
plt.legend()
plt.grid(True)
plt.show()