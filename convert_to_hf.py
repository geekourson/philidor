"""Convertit un checkpoint du projet au format HuggingFace `LlamaForCausalLM`.

Pourquoi Llama, alors qu'on n'a rien copié de Llama ? Parce qu'en choisissant
séparément RMSNorm, RoPE, SwiGLU et le pre-norm, quatre choix qui se
justifient chacun indépendamment, on a reconstruit exactement l'architecture
de Llama. La correspondance n'est donc pas une approximation : c'est le même
calcul, écrit autrement.

Ce que ça débloque, et c'est l'essentiel :

* le modèle se charge avec `transformers` sans une ligne de notre code ;
* il se publie sur HuggingFace comme n'importe quel modèle Llama ;
* et surtout, `convert_hf_to_gguf.py` de llama.cpp sait le convertir en GGUF,
  donc le modèle devient utilisable par `llama-server`, `llama-cli`, Ollama,
  LM Studio et tout ce qui parle GGUF.

Le script vérifie systématiquement que les deux implémentations produisent les
mêmes logits, à la précision flottante près. Une conversion qui « a l'air de
marcher » mais décale d'un demi-pourcent produirait un modèle silencieusement
dégradé, le genre d'erreur qu'on ne remarque qu'en comparant les Elo.

Usage :
    python convert_to_hf.py --ckpt checkpoints/run1_best.pt --out hf/chess-51m
"""

import argparse
import json
import os
import shutil

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from model import ChessGPT, ModelConfig


def build_state_dict(src: dict, cfg: ModelConfig) -> dict:
    """Renomme et redécoupe les poids vers la nomenclature de Llama.

    Une seule opération est non triviale : notre projection q/k/v est une
    matrice unique de taille 3*d x d, fusionnée pour n'avoir qu'un gros produit
    matriciel au lieu de trois petits. Llama garde les trois séparées, il faut
    donc la redécouper, dans le bon ordre, qui est celui du `split` de notre
    passe avant.
    """
    d = cfg.n_embd
    out = {}
    out["model.embed_tokens.weight"] = src["tok_emb.weight"]

    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        q = f"model.layers.{i}."
        qkv = src[p + "attn.qkv.weight"]
        wq, wk, wv = qkv.split(d, dim=0)
        out[q + "self_attn.q_proj.weight"] = wq
        out[q + "self_attn.k_proj.weight"] = wk
        out[q + "self_attn.v_proj.weight"] = wv
        out[q + "self_attn.o_proj.weight"] = src[p + "attn.proj.weight"]
        out[q + "mlp.gate_proj.weight"] = src[p + "mlp.gate.weight"]
        out[q + "mlp.up_proj.weight"] = src[p + "mlp.up.weight"]
        out[q + "mlp.down_proj.weight"] = src[p + "mlp.down.weight"]
        out[q + "input_layernorm.weight"] = src[p + "norm_attn.weight"]
        out[q + "post_attention_layernorm.weight"] = src[p + "norm_mlp.weight"]

    out["model.norm.weight"] = src["norm_final.weight"]
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = src["lm_head.weight"]
    return out


def make_tokenizer_files(vocab_path: str, out_dir: str):
    """Écrit un tokenizer au niveau du mot, où un « mot » est un coup UCI.

    C'est le point le plus inhabituel de cette conversion. Les modèles de
    langage courants découpent le texte en fragments de mots ; ici le
    vocabulaire est fermé, 1971 entrées, une par coup possible, et la
    séparation se fait sur les espaces. Le format `WordLevel` de la
    bibliothèque `tokenizers` correspond exactement à ce besoin.

    Conséquence pratique : le modèle se pilote en écrivant les coups séparés
    par des espaces, par exemple « e2e4 e7e5 g1f3 », et il complète la suite.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    with open(vocab_path) as f:
        v = json.load(f)
    stoi = v["stoi"]

    tok = Tokenizer(models.WordLevel(vocab=stoi, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # Le décodeur doit rejoindre les coups par des espaces. `Fuse` les
    # concatène sans séparateur, ce qui produit une chaîne du type
    # « e2e4e7e5g1f3 » indécodable : les coups UCI faisant quatre ou cinq
    # caractères, on ne peut plus les redécouper de façon fiable.
    # `WordPiece` avec un préfixe absent de notre vocabulaire fait exactement
    # ce qu'il faut : il joint par des espaces sans rien retirer.
    tok.decoder = decoders.WordPiece(prefix=chr(32), cleanup=False)
    tok.save(os.path.join(out_dir, "tokenizer.json"))

    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 256,
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<pad>",
            "clean_up_tokenization_spaces": False,
        }, f, indent=2)

    with open(os.path.join(out_dir, "special_tokens_map.json"), "w") as f:
        json.dump({"bos_token": "<bos>", "eos_token": "<eos>",
                   "pad_token": "<pad>", "unk_token": "<pad>"}, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--out", required=True)
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="écart maximal toléré sur les logits")
    args = p.parse_args()

    from transformers import LlamaConfig, LlamaForCausalLM

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    print(f"source : {args.ckpt}")
    print(f"  {cfg.n_layer} couches, d={cfg.n_embd}, {cfg.n_head} têtes, "
          f"MLP {cfg.mlp_hidden}, vocabulaire {cfg.vocab_size}")

    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.n_embd,
        intermediate_size=cfg.mlp_hidden,
        num_hidden_layers=cfg.n_layer,
        num_attention_heads=cfg.n_head,
        num_key_value_heads=cfg.n_head,     # pas de GQA : autant de clés que de requêtes
        max_position_embeddings=cfg.block_size,
        rms_norm_eps=1e-6,                  # identique à notre RMSNorm
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_embeddings,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )

    hf = LlamaForCausalLM(hf_cfg)
    missing, unexpected = hf.load_state_dict(
        build_state_dict(ck["model"], cfg), strict=False)
    # Avec les embeddings liés, lm_head.weight n'est pas dans le dictionnaire :
    # transformers le relie lui-même à embed_tokens. C'est la seule absence
    # acceptable.
    assert not unexpected, f"poids inattendus : {unexpected}"
    assert all("lm_head" in m for m in missing), f"poids manquants : {missing}"
    hf.tie_weights()

    # --- Vérification numérique ---------------------------------------------
    ours = ChessGPT(cfg)
    ours.load_state_dict(ck["model"])
    ours.eval()
    hf.eval()

    torch.manual_seed(0)
    max_err = 0.0
    with torch.no_grad():
        for T in (1, 7, 64, cfg.block_size):
            x = torch.randint(0, cfg.vocab_size, (3, T))
            a, _ = ours(x)
            b = hf(x).logits
            err = (a - b).abs().max().item()
            max_err = max(max_err, err)
            print(f"  longueur {T:>3} : écart max sur les logits = {err:.2e}")

    if max_err > args.tolerance:
        raise SystemExit(
            f"\nECHEC : écart de {max_err:.2e}, au-dessus de la tolérance "
            f"{args.tolerance:.0e}. La conversion est incorrecte, ne pas publier.")
    print(f"\n[OK] écart maximal {max_err:.2e}, les deux implémentations "
          f"calculent la même chose.")

    # --- Écriture -----------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    hf.save_pretrained(args.out, safe_serialization=True)
    make_tokenizer_files(args.vocab, args.out)
    shutil.copy(args.vocab, os.path.join(args.out, "vocab_uci.json"))

    n = sum(p.numel() for p in hf.parameters())
    print(f"\nÉcrit dans {args.out}/ ({n/1e6:.1f} M paramètres)")
    for f in sorted(os.listdir(args.out)):
        sz = os.path.getsize(os.path.join(args.out, f))
        print(f"  {f:<32} {sz/1e6:8.2f} Mo")


if __name__ == "__main__":
    main()
