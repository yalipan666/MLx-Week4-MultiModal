import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from RandomDigitsOnCanvas import RandomDigitsOnCanvas
from dataclasses import dataclass


# Set random seed for reproducibility
torch.manual_seed(42)

# pre-set all the relevant parameters
@dataclass
class TrainingHyperparameters:
    batch_size: int = 160
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    drop_rate: float = 0.1

@dataclass
class ModelHyperparameters:
    img_width: int = 280
    img_channels: int = 1
    num_classes: int = 10
    patch_size: int = 28
    embed_dim: int = 192
    num_heads: int = 6
    num_layers: int = 4
    n_digit: int = 10
    padding_token: int = 12  # Add a padding token index

# Define the encoder architecture
class ViT(nn.Module):
    def __init__(self, model_cfg: ModelHyperparameters, train_cfg: TrainingHyperparameters):
        super().__init__() #call the parent class's __init__
        ##### building the encoder
        # get the CLS which "summerize" the information of the whole sequence
        # the use of nn.Parameter here will garantee a correct backpop and update parameters
        self.cls = nn.Parameter(torch.randn(1,1,model_cfg.embed_dim))
        # get the embedding layer
        self.emb = nn.Linear(model_cfg.img_channels*model_cfg.patch_size*model_cfg.patch_size, model_cfg.embed_dim) #will do the broadcast to the data tensor, so the input and output dim don't need to be matched
        # get the positional encoding [including the patch_embeddings plus 1--the cls token]
        num_patches = (model_cfg.img_width // model_cfg.patch_size) * (model_cfg.img_width // model_cfg.patch_size)
        self.pos = nn.Embedding(num_patches + 1, model_cfg.embed_dim)
        # only register the parameters
        self.register_buffer('rng', torch.arange(num_patches + 1))
        # build the encoder, first define the encoder_layer, then just stack them together num_layers times
        # the use of mudulelist here will garantee a correct backpop and update parameters
        self.enc = nn.ModuleList([EncoderLayer(model_cfg.embed_dim, model_cfg.num_heads, train_cfg.drop_rate) for _ in range(model_cfg.num_layers)])
        # define the finnal output layer in the model
        self.fin = nn.Sequential(
            nn.LayerNorm(model_cfg.embed_dim),
            nn.Linear(model_cfg.embed_dim, model_cfg.num_classes)
        )
    
        #### building the decoder
        # +2 for start and end tokens
        vocab_size = model_cfg.num_classes + 3  # 10 digits + start + end + padding
        self.emb_output = nn.Embedding(vocab_size, model_cfg.embed_dim, padding_idx=model_cfg.padding_token)
        self.pos_output = nn.Embedding(model_cfg.n_digit+2, model_cfg.embed_dim)
        self.register_buffer('rng_output', torch.arange(model_cfg.n_digit+2))
        self.dec = nn.ModuleList([DecoderLayer(model_cfg.embed_dim, model_cfg.num_heads, train_cfg.drop_rate) for _ in range(model_cfg.num_layers)])
        self.fin_output = nn.Sequential(
            nn.LayerNorm(model_cfg.embed_dim),
            nn.Linear(model_cfg.embed_dim, vocab_size)
        )
        self.n_digit = model_cfg.n_digit
        self.num_classes = model_cfg.num_classes
        self.start_token = model_cfg.num_classes
        self.end_token = model_cfg.num_classes + 1
        self.padding_token = model_cfg.padding_token

    def forward(self, x, y):
        # flatten the patch matrix into a vector, also stack together all patches
        
        ### full-transformer
        b, np, ph, pw = x.shape  #[128,16,14,14]
        # flatte each patch into one vector
        x = x.reshape(b, np, ph*pw)  #[128, 16, 196]
        
        # each flatten patch will be embedded to the embed_dim
        pch = self.emb(x)

        # pre-pend the CLS ("secretory") token in front of the embedding
        cls = self.cls.expand(b,-1,-1) #CLS token for the whole batch
        hdn = torch.cat([cls, pch], dim=1)
        
        # add the position embeddings, which are learnable parameters
        hdn = hdn + self.pos(torch.arange(hdn.size(1), device=hdn.device)) # "broadcast" the positional encoding across the batch dimension.
        # go into the transformer attention blocks --- the encoder
        for enc in self.enc: hdn = enc(hdn)  # the key and val of the encoder goes into the decoder
        # # this section is only useful for the classification task
        # # only select the hidden vector of the CLS token for making the prediction
        # out = hdn[:,0,:]
        # # go throught the finnal fc layer for the classification task
        # out = self.fin(out)
        ##### build the decoder, using y(i.e.,target), a list of 6 elements [start,digit1,digit2,digit3,digit4,end]
    
        out_emb = self.emb_output(y) #[batch, seq_len, embed_dim]
        out_emb = out_emb + self.pos_output(self.rng_output[:out_emb.size(1)])
        tgt_mask = generate_mask(out_emb.size(1)).to(x.device)
        tgt = out_emb
        for dec in self.dec: tgt = dec(tgt, memory=hdn, tgt_mask=tgt_mask)
        # memory indicates the output from the encoder layer, which will be projected by the decoder's 
        # cross-attention layer into keys and values as needed
        
        # get the finnal prediction
        logits = self.fin_output(tgt)
        return logits

    # build the generation task where the model predict the token/digit one by one
    def autoregressive_inference(self, x, device, max_digits=None): 
        # patchify and encode the image
        # x: [batch, np, ph, pw]
        b, np, ph, pw = x.shape
        x = x.reshape(b, np, ph*pw)
        pch = self.emb(x)
        cls = self.cls.expand(b,-1,-1)
        hdn = torch.cat([cls, pch], dim=1)
        hdn = hdn + self.pos(self.rng)
        for enc in self.enc: hdn = enc(hdn)
        # Initialize the output sequence, start with [start_token]
        max_steps = max_digits if max_digits is not None else self.n_digit + 2
        y = torch.full((b, 1), self.start_token, dtype=torch.long, device=device) # the "have been seen" tokens
        outputs = []
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for step in range(max_steps):
            out_emb = self.emb_output(y)
            out_emb = out_emb + self.pos_output(self.rng_output[:out_emb.size(1)])
            tgt_mask = generate_mask(out_emb.size(1)).to(x.device) # the mask is auto-adjusted to how many y you have here
            tgt = out_emb
            for dec in self.dec: # go throught the loop of the decoder layers
                tgt = dec(tgt, memory=hdn, tgt_mask=tgt_mask)
            logits = self.fin_output(tgt)  # [b, cur_seq, num_classes]
            next_token_logits = logits[:, -1, :]  # [b, num_classes]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [b, 1] Returns the indices of the maximum value of all elements in the input tensor.
            # If already finished, keep predicting end_token
            next_token[finished.unsqueeze(1)] = self.end_token
            y = torch.cat([y, next_token], dim=1)
            outputs.append(next_token)
            # Update finished mask
            finished = finished | (next_token.squeeze(1) == self.end_token)
            # If all finished, break
            if finished.all():
                break
        # Stack outputs, remove start token, and trim at end_token for each sample
        outputs = torch.cat(outputs, dim=1)  # [b, <=max_steps]
        # Remove tokens after end_token for each sample
        result = []
        for i in range(b):
            out = outputs[i]
            if (out == self.end_token).any():
                idx = (out == self.end_token).nonzero(as_tuple=True)[0][0]
                result.append(out[:idx].cpu())
            else:
                result.append(out.cpu())
        # Pad to max length in batch
        maxlen = max([r.size(0) for r in result])
        result_padded = torch.full((b, maxlen), self.end_token, dtype=torch.long) #long: only integers no decimals
        for i, r in enumerate(result):
            result_padded[i, :r.size(0)] = r
        return result_padded.to(device)

class EncoderLayer(nn.Module):
    def __init__(self,dim,num_heads,drop_rate):
        super().__init__()
        self.att = Attention(dim,num_heads,drop_rate)
        self.ini = nn.LayerNorm(dim)
        self.ffn = FFN(dim,drop_rate)
        self.fin = nn.LayerNorm(dim)

    def forward(self,src):
        # the skip connections in the residual block (residual:the diff between the input and the output-->the delta)
        out = self.att(src)
        src = src + out
        src = self.ini(src)
        out = self.ffn(src)
        src = src + out
        src = self.fin(src)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_rate):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, drop_rate)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self,tgt, memory, tgt_mask=None, memory_mask=None):
        # masked self-attention
        tgt2,_ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        # cross attention between encoder and decoder
        tgt2,_ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        # feed-forward
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

class FFN(nn.Module):
    def __init__(self,dim,drop_rate):
        super().__init__()
        self.one = nn.Linear(dim, dim)
        self.drp = nn.Dropout(drop_rate)
        self.rlu = nn.GELU()
        # self.rlu = nn.ReLU(inplace=True)
        self.two = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.one(x)
        x = self.rlu(x)
        x = self.drp(x)
        x = self.two(x)
        return x


class Attention(nn.Module): #MultiHeadAttention
    def __init__(self,dim,num_heads,drop_rate):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.drpout = nn.Dropout(drop_rate)

    def forward(self,x):
        B, N, C = x.shape #(batch, seq_len, embed_dim)
        # project and split into heads
        qry = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2) #(B, num_heads, N, head_dim)
        key = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2) 
        val = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2) 
        # scaled dot-product attention
        att = (qry @ key.transpose(-2,-1) / self.head_dim ** 0.5)
        att = torch.softmax(att, dim=-1)
        att = self.drpout(att)
        out = torch.matmul(att,val) #(B, num_heads, N, head_dim)
        # concatenate heads
        out = out.transpose(1,2).reshape(B, N, C) # (B, N, embed_dim)
        out = self.o_proj(out)
        # concatenate k and v
        key = key.transpose(1,2).reshape(B, N, C)
        val = val.transpose(1,2).reshape(B, N, C)
        return out


def patchify(batch_data, patch_size):
    """
    patchify the batch of images
    """
    b,c,h,w = batch_data.shape  #[batch_size,channels,height,width] 
    ph = patch_size
    pw = patch_size
    nh, nw = h//ph, w//pw

    batch_patches = torch.reshape(batch_data, (b,c,nh,ph,nw,pw))
    batch_patches = torch.permute(batch_patches,(0,1,2,4,3,5)) #[64,1,4,4,7,7]

    # flatten the pixels in each patch
    return batch_patches

def generate_mask(sz):
    # generate a (sz,sz) mask (upper triangle) with -inf above the diagonal, 0 elsewhere
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def train_model(model, train_loader, criterion, optimizer, device, model_cfg):
    model.train()
    running_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    for batch_idx, batch in enumerate(train_loader):
        canvas = batch[0]  # [batch, 1, H, W]
        target = batch[1]  # [batch, seq_len]
        # Patchify the canvas images
        data = patchify(canvas, model_cfg.patch_size)  # [batch, 1, nh, nw, ph, pw]
        b, c, nh, nw, ph, pw = data.shape
        data = data.reshape(b, c, nh * nw, ph, pw)  # [batch, 1, n_patches, ph, pw]
        data = data.squeeze(1)  # [batch, n_patches, ph, pw]
        data, target = data.to(device), target.to(device)
        # Pad target to max length in batch
        max_len = target.size(1)
        input_seq = target[:, :-1]
        target_seq = target[:, 1:]
        # ### try to plot a random image to have a peek
        # img = data[0].cpu().squeeze()
        # plt.imshow(img, cmap='gray')
        # plt.show()
        
        optimizer.zero_grad()
        output = model(data, input_seq)  # output: [batch, seq_len, vocab_size]
        output = output.reshape(-1, output.size(-1))
        target_seq = target_seq.reshape(-1)
        # Use ignore_index for padding
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Token-level accuracy (ignore padding)
        pred_tokens = output.argmax(dim=-1)
        mask = (target_seq != model_cfg.padding_token)
        correct_tokens += ((pred_tokens == target_seq) & mask).sum().item()
        total_tokens += mask.sum().item()
        if batch_idx % 100 == 99:
            print(f'Batch: {batch_idx + 1}, Loss: {running_loss/100:.3f}, Token Accuracy: {100.*correct_tokens/max(total_tokens,1):.2f}%')
            running_loss = 0.0
            correct_tokens = 0
            total_tokens = 0
        torch.cuda.empty_cache()

def evaluate_model(model, test_loader, device, model_cfg):
    model.eval()
    total_seqs = 0
    correct_seqs = 0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for batch in test_loader:
            canvas = batch[0]
            target = batch[1]
            data = patchify(canvas, model_cfg.patch_size)
            b, c, nh, nw, ph, pw = data.shape
            data = data.reshape(b, c, nh * nw, ph, pw)
            data = data.squeeze(1)
            data, target = data.to(device), target.to(device)
            target_seq = target[:, 1:]
            pred_seq = model.autoregressive_inference(data, device, max_digits=target_seq.size(1))
            min_len = min(target_seq.size(1), pred_seq.size(1))
            target_trim = target_seq[:, :min_len]
            pred_trim = pred_seq[:, :min_len]
            mask = (target_trim != model_cfg.padding_token)
            correct_tokens += ((pred_trim == target_trim) & mask).sum().item()
            total_tokens += mask.sum().item()
            correct_seqs += (((pred_trim == target_trim) | ~mask).all(dim=1)).sum().item()
            total_seqs += target_seq.size(0)
    token_acc = 100. * correct_tokens / max(total_tokens,1)
    seq_acc = 100. * correct_seqs / max(total_seqs,1)
    print(f'Test Token Accuracy: {token_acc:.2f}%, Sequence Accuracy: {seq_acc:.2f}%')
    return seq_acc

class SyntheticMNISTSequenceDataset(Dataset):
    """
    Loads synthetic MNIST images and variable-length digit sequences, pads them to n_digit,
    and adds start/end tokens for transformer training.
    """
    def __init__(self, pt_path, n_digit, start_token, end_token, padding_token):
        super().__init__()
        data = torch.load(pt_path)
        self.images = data[0]  # shape: [N, 1, H, W]
        self.labels = data[1]  # list of 1D tensors (variable length)
        self.n_digit = n_digit
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        # Pad label to n_digit
        label = label[:self.n_digit]  # truncate if too long
        pad_len = self.n_digit - len(label)
        if pad_len > 0:
            label = torch.cat([label, torch.full((pad_len,), self.padding_token, dtype=torch.long)])
        # Add start and end tokens
        label = torch.cat([
            torch.tensor([self.start_token], dtype=torch.long),
            label,
            torch.tensor([self.end_token], dtype=torch.long)
        ])  # shape: [n_digit + 2]
        return img, label

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # loadin the config
    train_cfg = TrainingHyperparameters()
    model_cfg = ModelHyperparameters()

    # Use the synthetic datasets
    train_dataset = SyntheticMNISTSequenceDataset(
        pt_path='synthetic_mnist_train.pt',
        n_digit=model_cfg.n_digit,
        start_token=model_cfg.num_classes,
        end_token=model_cfg.num_classes + 1,
        padding_token=model_cfg.padding_token
    )
    test_dataset = SyntheticMNISTSequenceDataset(
        pt_path='synthetic_mnist_test.pt',
        n_digit=model_cfg.n_digit,
        start_token=model_cfg.num_classes,
        end_token=model_cfg.num_classes + 1,
        padding_token=model_cfg.padding_token
    )
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # get the image dimension
    tmp = train_dataset[0]
    model_cfg.img_width = tmp[0].shape[1]  # canvas_tensor shape: (1, H, W)
    model_cfg.img_channels = 1
    model_cfg.patch_size = 14  # or set as needed

    # Initialize model, loss function, and optimizer
    model = ViT(model_cfg, train_cfg).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=model_cfg.padding_token)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    # Training loop
    print('Starting training...')
    for epoch in range(train_cfg.num_epochs):
        print(f'\nEpoch {epoch + 1}/{train_cfg.num_epochs}')
        train_model(model, train_loader, criterion, optimizer, device, model_cfg)
        evaluate_model(model, test_loader, device, model_cfg)

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_transformer_model.pth')
    print('Model saved to mnist_transformer_model.pth')

if __name__ == '__main__':
    main() 