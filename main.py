import torch
import numpy as np
from log import Logger

CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "head_dim": 64,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "mlp_hidd_dim": 4 * 384,
    "verbose": True,
    "batch_size": 32,
    "device": "cuda:1",
    "lr": 4e-4,
    "steps": 10000
}

class LayerNorm(torch.nn.Module):
    def __init__(self, cfg, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.shift = torch.nn.Parameter(torch.zeros(cfg["emb_dim"]))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["emb_dim"], cfg["mlp_hidd_dim"]),
            torch.nn.GELU(),
            torch.nn.Linear(cfg["mlp_hidd_dim"], cfg["emb_dim"])
        )
        
    def forward(self, x):
        return self.layers(x)
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Wq = torch.nn.Linear(cfg["emb_dim"], cfg["n_heads"]*cfg["head_dim"], bias=cfg["qkv_bias"])
        self.Wk = torch.nn.Linear(cfg["emb_dim"], cfg["n_heads"]*cfg["head_dim"], bias=cfg["qkv_bias"])
        self.Wv = torch.nn.Linear(cfg["emb_dim"], cfg["n_heads"]*cfg["head_dim"], bias=cfg["qkv_bias"])
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])
        self.out_proj = torch.nn.Linear(cfg["n_heads"] * cfg["head_dim"], cfg["emb_dim"])
        self.register_buffer('mask', torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1))
    
    def forward(self, x):
        B, T, C = x.shape
        
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        
        queries = queries.view(B, T, self.cfg["n_heads"], self.cfg["head_dim"])
        keys = keys.view(B, T, self.cfg["n_heads"], self.cfg["head_dim"])
        values = values.view(B, T, self.cfg["n_heads"], self.cfg["head_dim"])
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        
        mask_bool = self.mask.bool()[:T, :T]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / self.cfg["head_dim"] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        context_vec = context_vec.contiguous().view(B, T, self.cfg["n_heads"] * self.cfg["head_dim"])
        context_vec = self.out_proj(context_vec)

        return context_vec
    
class Block(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.mlp = MLP(cfg)
        self.ln_1 = LayerNorm(cfg)
        self.ln_2 = LayerNorm(cfg)
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x

class GPT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])
        
        self.blocks = torch.nn.Sequential(*[Block(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg)
        # The unembedding matrix Wu
        self.out_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)
        
        if cfg["verbose"]:
            self.print_info()
        
    def forward(self, x):
        B, T = x.shape
        
        tok_embs = self.tok_emb(x)
        pos_embs = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_embs + pos_embs
        x = self.drop_emb(x)
        
        x = self.blocks(x)
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -self.cfg["context_length"]:]
            
            logits = self(idx_cond)
                
            logits = logits[:, -1, :] / temperature
            
            if top_k != None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probas = torch.softmax(logits, dim=-1)
            
            ## Greedy Sampling
#             idx_next = torch.argmax(probas, dim=-1, keepdim=True)

            idx_next = torch.multinomial(probas, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    
    def print_info(self):
        
        print(f"Initialzed a new model with config:")
        for conf, val in self.cfg.items():
            print(f"     {conf}: {val}")        

        # Calculate the total parameters in the model
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters in the model: {total_params/1e6:.2f} Million")
        
        # Assume float32 which takes 4 bytes a patameter
        total_model_size = total_params * 4 / (1024 * 1024)
        print(f"Total size of the model: {total_model_size:.2f} MB")

with open("./train.txt", "r") as f:
    data = f.read()
f.close()
print(f"Total length of the dataset in characters: {len(data)}")

vocab = sorted(list(set(data)))
print("Vocabulary: ", "".join(vocab))
print(f"Total length of vocabulary: {len(vocab)}")
itos = {i: char for i, char in enumerate(vocab)}
stoi = {char: i for i, char in enumerate(vocab)}

encode = lambda s: [stoi[char] for char in s]
decode = lambda tkns: "".join([itos[i] for i in tkns])

CONFIG["vocab_size"] = len(vocab)
model = GPT(CONFIG)
model.to(CONFIG["device"])

model.eval()

text = "colonel jose aureliano buendia"
result = model.generate(torch.tensor([encode(text)], device=CONFIG["device"]), 100)
print(decode(result.squeeze().detach().cpu().tolist()))

data = np.array(encode(data))

def get_batch():

    ix = torch.randint(len(data) - CONFIG["context_length"], (CONFIG["batch_size"],))
    x = torch.stack([torch.from_numpy((data[i:i+CONFIG["context_length"]]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+CONFIG["context_length"]]).astype(np.int64)) for i in ix])
    if CONFIG["device"] == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(CONFIG["device"], non_blocking=True), y.pin_memory().to(CONFIG["device"], non_blocking=True)
    else:
        x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
    return x, y

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
step = 0
log = Logger("gpt0.1", model.cfg, 2)
while step < model.cfg["steps"]:
    
    x, y = get_batch()
    
    model.train()
    optimizer.zero_grad()
    
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    print(step, loss.item())
    log([step, loss.item()])
    
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        text = "casting the search for truth"
        result = model.generate(torch.tensor([encode(text)], device=CONFIG["device"]), 1000)
        print(decode(result.squeeze().detach().cpu().tolist()))
        
    step += 1