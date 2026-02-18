#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import numpy as np
import jittor as jt
import jittor.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import glob
jt.flags.use_cuda=1
if hasattr(jt.flags, "opt_level"):
    jt.flags.opt_level=0
jt.flags.log_silent=1
if "JT_SYNC" in os.environ:
    del os.environ["JT_SYNC"]
os.environ["JT_SAVE_MEM"]="1"
os.environ["cuda_arch"]="0" 
os.environ["use_cutt"]="0"
hf_token=os.environ["HF_TOKEN"]
print(os.environ.get("HF_TOKEN"))




class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5): 
        super().__init__()
        """默认float32"""
        self.weight=jt.ones(dim) 
        self.eps=eps

    def execute(self, x):
        """严格在 float32 下计算 norm"""
        x_f32=x.float32()
        variance=jt.mean(x_f32 * x_f32, dim=-1, keepdims=True)
       
        normed=x_f32 * jt.rsqrt(variance + self.eps)
        """转换回输入类型再乘以权重"""
        return (normed.cast(x.dtype)) * self.weight.cast(x.dtype)

def rotate_half(x):
    x1, x2=x.chunk(2, dim=-1)
    return jt.concat([-x2, x1], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_pos=131072, base=500000.0, loaded_inv_freq=None):
        super().__init__()
        
        """如果传入了从 PyTorch 导出的 inv_freq，直接使用"""
        if loaded_inv_freq is not None:
            self.inv_freq=loaded_inv_freq
        else:
            """只有在没加载到时才退回默认计算，作为保底预防"""
            print("Warning: Computing inv_freq manually. Accuracy may drop for Llama 3.2.")
            self.inv_freq=1.0 / (base ** (jt.arange(0, dim, 2).float32() / dim))
            
      
        t=jt.arange(max_pos).float32()
        freqs=jt.outer(t, self.inv_freq) 
        
        
        emb=jt.concat([freqs, freqs], dim=-1)
        self.cos=jt.cos(emb).float16()
        self.sin=jt.sin(emb).float16()

    def get_embed(self, seq_len):
        return self.cos[:seq_len], self.sin[:seq_len]

def apply_rotary_pos_emb(q, k, cos, sin):
    """q, k: [B, T, H, D]"""
    """cos, sin: [T, D] -> 需要扩展为 [1, T, 1, D]"""

    """ 显式 unsqueeze"""
    cos=cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
    sin=sin.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
    if cos.dtype != q.dtype:
        cos=cos.astype(q.dtype)
        sin=sin.astype(q.dtype)

    q_embed=(q * cos) + (rotate_half(q) * sin)
    k_embed=(k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaAttention(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.hidden_size=config["hidden_size"]
        self.num_heads=config["num_attention_heads"]
        self.num_kv_heads=config["num_key_value_heads"]
        self.head_dim=self.hidden_size // self.num_heads
        self.num_kv_groups=self.num_heads // self.num_kv_heads
        self.rotary=rotary

        use_bias=config.get("attention_bias", False)
        self.q_proj=nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=use_bias)
        self.k_proj=nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj=nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj=nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def execute(self, x):
        B, T, C=x.shape

        """投影并分头"""
        q=self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k=self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v=self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        """应用 RoPE"""
        cos, sin=self.rotary.get_embed(T)
        q, k=apply_rotary_pos_emb(q, k, cos, sin)

        """GQA 展开 (K/V 复制)"""
        if self.num_kv_groups > 1:
            k=k.unsqueeze(3).repeat(1, 1, 1, self.num_kv_groups, 1).reshape(B, T, self.num_heads, self.head_dim)
            v=v.unsqueeze(3).repeat(1, 1, 1, self.num_kv_groups, 1).reshape(B, T, self.num_heads, self.head_dim)

        """转换维度用于矩阵乘法: (B, H, T, D)"""
        q=q.transpose(0, 2, 1, 3)
        k=k.transpose(0, 2, 1, 3)
        v=v.transpose(0, 2, 1, 3)

        """关键：精度保护，在 Float32 下计算 Attention Score"""

        q_f32=q.float32()
        k_f32=k.float32()


        attn=jt.matmul(q_f32, k_f32.transpose(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        """关键：严格 Mask"""
        """使用一个非常大的负数，确保 Softmax 后彻底为 0"""
        mask=jt.triu(jt.ones((T, T)), diagonal=1).bool()
        mask=mask.unsqueeze(0).unsqueeze(0) 

       
        attn=jt.where(mask, jt.array(-1e10).astype(attn.dtype), attn)

       
        attn=nn.softmax(attn, dim=-1)

        """乘回 V 并转回 FP16 (如果 V 是 FP16)"""
       
        out=jt.matmul(attn, v.float32()).cast(x.dtype)

        
        out=out.transpose(0, 2, 1, 3).reshape(B, T, self.hidden_size)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj=nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj=nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj=nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def execute(self, x):
        x_f32=x.float32()
        """ 手动写 matmul 避开 nn.Linear 可能触发的 FP16 融合 Bug"""
        gate=jt.matmul(x_f32, self.gate_proj.weight.float32().transpose())
        up=jt.matmul(x_f32, self.up_proj.weight.float32().transpose())

        intermediate=nn.silu(gate) * up

        down=jt.matmul(intermediate.float32(), self.down_proj.weight.float32().transpose())
        return down.cast(x.dtype)
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.input_layernorm=RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.self_attn=LlamaAttention(config, rotary)
        self.post_attention_layernorm=RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.mlp=LlamaMLP(config)

    def execute(self, x):
        x=x + self.self_attn(self.input_layernorm(x))
        x=x + self.mlp(self.post_attention_layernorm(x))
        return x

class Llama3B(nn.Module):
    def __init__(self, config, inv_freq_data=None): 
        super().__init__()
        self.config=config
        self.embed_tokens=nn.Embedding(config["vocab_size"], config["hidden_size"])

        head_dim=config["hidden_size"] // config["num_attention_heads"]
        
        """将 inv_freq 传入"""
        self.rotary=RotaryEmbedding(
            head_dim, 
            max_pos=config["max_position_embeddings"], 
            base=config["rope_theta"],
            loaded_inv_freq=inv_freq_data
        )

        """传递 rotary 给每一层"""
        self.layers=nn.ModuleList([LlamaDecoderLayer(config, self.rotary) for _ in range(config["num_hidden_layers"])])
        self.norm=RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        """初始化 lm_head """
        self.lm_head=nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

 
    def execute(self, input_ids):
        x=self.embed_tokens(input_ids)
        for layer in self.layers:
            x=layer(x)
        x=self.norm(x)
    
        logits=jt.matmul(x.float32(), self.lm_head.weight.float32().transpose())
    
        
        return logits





# In[21]:


def save_pt_to_jt_shards(pt_model, save_dir, layers_per_shard=4):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Converting and saving weights to {save_dir}...")

    """提取 RoPE 的 inv_freq """
    """只要取第一层的即可，所有层都一样"""
    inv_freq=pt_model.model.layers[0].self_attn.rotary_emb.inv_freq
    inv_freq_np=inv_freq.detach().cpu().numpy().astype(np.float32)
    """单独保存 inv_freq"""
    np.savez(os.path.join(save_dir, "inv_freq.npz"), inv_freq=inv_freq_np)
    print("Saved special RoPE inv_freq.")

    """常规权重保存，保持原有逻辑"""
    state_dict=pt_model.state_dict()
    shards={}
    
    for k, v in state_dict.items():
        """依然跳过 rotary_emb ，因为已经手动保存了 inv_freq"""
        if "rotary_emb" in k: 
            continue

        w=v.detach().cpu().numpy().astype(np.float16)
        

        if "layers." in k:
            try:
                layer_idx=int(k.split("layers.")[1].split(".")[0])
                shard_id=(layer_idx // layers_per_shard) + 1
            except:
                shard_id=0
        
        shard_file=f"shard_{shard_id:03d}.npz"
        if shard_file not in shards:
            shards[shard_file]={}
        shards[shard_file][k]=w

    """ 处理lm_head"""
    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
         pass 

    for filename, data in shards.items():
        save_path=os.path.join(save_dir, filename)
        np.savez(save_path, **data)
    
    print("All shards saved.")
    del state_dict
    gc.collect()


# In[15]:


def load_weights_correctly(model, shard_dir):
    print(f"Loading weights from {shard_dir}...")
    import glob
    shard_files=sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))

    model_sd=model.state_dict()
    cnt=0

    for shard_path in shard_files:
        print(f"  -> Processing {os.path.basename(shard_path)}...")
        data=np.load(shard_path, allow_pickle=True)

        for k_file in data.files:
            k_jt=k_file
            if k_file.startswith("model."):
                k_jt=k_file.replace("model.", "", 1)
            if "lm_head" in k_jt:
                continue

            if k_jt not in model_sd:
                continue

            w=data[k_file]
            if model_sd[k_jt].shape!=w.shape:
                if model_sd[k_jt].shape==w.T.shape:
                    w=w.T
                else:
                    print(f"Shape mismatch: {k_jt} model:{model_sd[k_jt].shape} vs file:{w.shape}")
                    continue

            
            model_sd[k_jt].assign(w)
            cnt += 1

        del data
        jt.gc()

    print(f"Loaded {cnt} tensors. Model is ready.")


# In[31]:


def initialize_jittor_model(model_name,config):

    save_model_name=model_name
    if '/' in save_model_name:
        save_model_name=save_model_name.replace('/','___OF_')
    shard_dir=f'/mnt/d/code/Artificial_Intelligence/{save_model_name}'
    tokenizer_path=f'/mnt/d/code/Artificial_Intelligence/{save_model_name}/tokenizer'
    
       
    
    inv_freq_path=os.path.join(shard_dir, "inv_freq.npz")
    loaded_inv_freq=None
    if os.path.exists(inv_freq_path):
        print(f"Loading custom inv_freq from {inv_freq_path}")
        loaded_inv_freq=jt.array(np.load(inv_freq_path)["inv_freq"])
    
    if os.path.exists(shard_dir):
        print('Loading from local...')
        tokenizer=AutoTokenizer.from_pretrained(model_name)
        
        
        model=Llama3B(config, inv_freq_data=loaded_inv_freq)
        model.eval()

        load_weights_correctly(model, shard_dir)
        
       
        model.lm_head.weight=model.embed_tokens.weight
        
        return model, tokenizer
    print('loading from hugging face')
    tokenizer=AutoTokenizer.from_pretrained(
        model_name, 
        token=hf_token
    )
    tokenizer.save_pretrained(tokenizer_path)

    pt_model=AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        token=hf_token,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    jt_shard_path=f'/mnt/d/code/Artificial_Intelligence/{save_model_name}'
    
    save_pt_to_jt_shards(pt_model, jt_shard_path)
    # model,tokenizer=initialize_jittor_model(model_name,config)
    print('No model,No tokenizer')
    return None,None

