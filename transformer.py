from pathlib import Path
import math
import time
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import (
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import MeCab
import gc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir_path = Path('model')
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)

file_paths_train = ['./train_data/train.original', './train_data/train.modern']
file_paths_valid = ['./val_data/val.original', './val_data/val.modern']

def read_texts(file_path):
    with open(file_path, 'r') as file:
        texts = file.readlines()
    return texts

texts_src_train = read_texts(file_paths_train[0])
texts_tgt_train = read_texts(file_paths_train[1])
texts_src_valid = read_texts(file_paths_valid[0])
texts_tgt_valid = read_texts(file_paths_valid[1])

for src, tgt in zip(texts_src_train[:3], texts_tgt_train[:3]):
    print(src.strip('\n'))
    print('↓')
    print(tgt.strip('\n'))
    print('')

tagger_original = MeCab.Tagger('-O wakati -d ./UniDic-wabun_1603/')
tagger_modern = MeCab.Tagger('-O wakati -d ./unidic-cwj-3.1.1-full/')

tokenizer_src = get_tokenizer(tagger_original.parse, language='ja')
tokenizer_tgt = get_tokenizer(tagger_modern.parse, language='ja')

def build_vocab(texts, tokenizer):

    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    return Vocab(counter, specials=['<unk>', '<pad>', '<start>', '<end>'])

vocab_src = build_vocab(texts_src_train, tokenizer_src)
vocab_tgt = build_vocab(texts_tgt_train, tokenizer_tgt)

for char, index in list(vocab_src.stoi.items())[:15]:
    print('Char: {: <8} → Index: {: <2}'.format(char, index))

def data_process(texts_src, texts_tgt, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt):

    data = []
    for (src, tgt) in zip(texts_src, texts_tgt):
        src_tensor = torch.tensor(
            convert_text_to_indexes(text=src, vocab=vocab_src, tokenizer=tokenizer_src),
            dtype=torch.long
        )
        tgt_tensor = torch.tensor(
            convert_text_to_indexes(text=tgt, vocab=vocab_tgt, tokenizer=tokenizer_tgt),
            dtype=torch.long
        )
        data.append((src_tensor, tgt_tensor))

    return data

def convert_text_to_indexes(text, vocab, tokenizer):
    return [vocab['<start>']] + [
        vocab[token] for token in tokenizer(text.strip("\n"))
    ] + [vocab['<end>']]

train_data = data_process(
    texts_src=texts_src_train, texts_tgt=texts_tgt_train,
    vocab_src=vocab_src, vocab_tgt=vocab_tgt,
    tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt
)
valid_data = data_process(
    texts_src=texts_src_valid, texts_tgt=texts_tgt_valid,
    vocab_src=vocab_src, vocab_tgt=vocab_tgt,
    tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt
)

print('インデックス化された文章')
print('Input: {}\nOutput: {}'.format(train_data[0][0], train_data[0][1]))
print('')

print('インデックス化された文章をもとに戻す')
print('Input: {}\nOutput: {}'.format(
    ' '.join([vocab_src.itos[x] for x in train_data[0][0]]),
    ' '.join([vocab_tgt.itos[x] for x in train_data[0][1]])
))

batch_size = 256
PAD_IDX = vocab_src['<pad>']
START_IDX = vocab_src['<start>']
END_IDX = vocab_src['<end>']

def generate_batch(data_batch):

    batch_src, batch_tgt = [], []
    for src, tgt in data_batch:
        batch_src.append(src)
        batch_tgt.append(tgt)

    batch_src = pad_sequence(batch_src, padding_value=PAD_IDX)
    batch_tgt = pad_sequence(batch_tgt, padding_value=PAD_IDX)

    return batch_src, batch_tgt

train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

list(train_iter)[0]

class Seq2SeqTransformer(nn.Module):

    def __init__(
        self, num_encoder_layers: int, num_decoder_layers: int,
        embedding_size: int, vocab_size_src: int, vocab_size_tgt: int,
        dim_feedforward:int = 512, dropout:float = 0.1, nhead:int = 8
    ):

        super(Seq2SeqTransformer, self).__init__()

        self.token_embedding_src = TokenEmbedding(vocab_size_src, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.token_embedding_tgt = TokenEmbedding(vocab_size_tgt, embedding_size)
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output = nn.Linear(embedding_size, vocab_size_tgt)

    def forward(
        self, src: Tensor, tgt: Tensor,
        mask_src: Tensor, mask_tgt: Tensor,
        padding_mask_src: Tensor, padding_mask_tgt: Tensor,
        memory_key_padding_mask: Tensor
    ):

        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src, padding_mask_src)
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(
            embedding_tgt, memory, mask_tgt, None,
            padding_mask_tgt, memory_key_padding_mask
        )
        return self.output(outs)

    def encode(self, src: Tensor, mask_src: Tensor):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt: Tensor, memory: Tensor, mask_tgt: Tensor):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_size):

        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)

class PositionalEncoding(nn.Module):

    def __init__(self, embedding_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.embedding_pos[: token_embedding.size(0), :])

def create_mask(src, tgt, PAD_IDX):

    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt, PAD_IDX)
    mask_src = torch.zeros((seq_len_src, seq_len_src), device=device).type(torch.bool)

    padding_mask_src = (src == PAD_IDX).transpose(0, 1)
    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)

    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt


def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))
    return mask

def train(model, data, optimizer, criterion, PAD_IDX):

    model.train()
    losses = 0
    for src, tgt in tqdm(data):

        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        logits = model(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )

        optimizer.zero_grad()

        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(data)

def evaluate(model, data, criterion, PAD_IDX):

    model.eval()
    losses = 0
    for src, tgt in data:

        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        logits = model(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )

        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        losses += loss.item()

    return losses / len(data)

vocab_size_src = len(vocab_src)
vocab_size_tgt = len(vocab_tgt)
embedding_size = 240
nhead = 8
dim_feedforward = 100
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.1

model = Seq2SeqTransformer(
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    embedding_size=embedding_size,
    vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt,
    dim_feedforward=dim_feedforward,
    dropout=dropout, nhead=nhead
)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(model.parameters())

epoch = 30
best_loss = float('Inf')
best_model = None
patience = 10
counter = 0

for loop in range(1, epoch + 1):

    start_time = time.time()
    gc.collect()

    loss_train = train(
        model=model, data=train_iter, optimizer=optimizer,
        criterion=criterion, PAD_IDX=PAD_IDX
    )

    elapsed_time = time.time() - start_time

    loss_valid = evaluate(
        model=model, data=valid_iter, criterion=criterion, PAD_IDX=PAD_IDX
    )

    print('[{}/{}] train loss: {:.2f}, valid loss: {:.2f}  [{}{:.0f}s] count: {}, {}'.format(
        loop, epoch,
        loss_train, loss_valid,
        str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
        elapsed_time % 60,
        counter,
        '**' if best_loss > loss_valid else ''
    ))

    if best_loss > loss_valid:
        best_loss = loss_valid
        best_model = model
        counter = 0


    counter += 1

torch.save(best_model.state_dict(), './save/main_30.pth')


