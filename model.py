"""Le modèle : un Transformer décodeur, écrit en PyTorch pur.

L'idée centrale du projet tient en une phrase : on ne code aucune règle du jeu.
Le modèle reçoit une suite de coups et doit prédire le suivant, exactement comme
un modèle de langage prédit le mot suivant. Il n'a pas d'échiquier en mémoire,
pas de notion de pièce, pas de vérificateur de légalité. S'il finit par jouer
des coups légaux, c'est uniquement parce que c'est le seul moyen de bien
prédire des parties jouées par des humains.

L'architecture est celle des modèles récents (Llama et ses descendants) plutôt
que celle du GPT-2 d'origine. Quatre choix, expliqués un par un plus bas :
pre-norm, RMSNorm, RoPE et SwiGLU.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 1971      # 1968 coups UCI + <pad> <bos> <eos>
    block_size: int = 256       # contexte, en coups
    n_layer: int = 16
    n_head: int = 8
    n_embd: int = 512
    mlp_hidden: int = 1408      # multiple de 64, ~8/3 x n_embd (voir SwiGLU)
    rope_theta: float = 10000.0
    dropout: float = 0.0        # on a bien plus de données que de paramètres
    tie_embeddings: bool = True


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Normalisation par la racine de la moyenne des carrés.

    La LayerNorm classique fait deux choses : elle recentre les valeurs autour
    de zéro, puis les remet à une échelle unitaire. La RMSNorm ne fait que la
    seconde. Il s'avère que le recentrage n'apporte presque rien en pratique,
    alors que le calculer coûte une passe supplémentaire sur les données.
    On garde donc uniquement la mise à l'échelle.

    L'intuition : ce qui compte pour la stabilité de l'entraînement, c'est que
    les valeurs qui circulent dans le réseau gardent une taille comparable
    d'une couche à l'autre. Savoir si elles sont centrées sur zéro ou sur trois
    est secondaire.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Le calcul se fait en float32 même quand x est en bf16 : une somme de
        # carrés sur 512 valeurs perd trop de précision en 16 bits.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


# ---------------------------------------------------------------------------
# RoPE : Rotary Position Embedding
# ---------------------------------------------------------------------------

def build_rope_cache(block_size: int, head_dim: int, theta: float, device, dtype):
    """Pré-calcule les cosinus et sinus de rotation pour chaque position.

    Comment un Transformer sait-il dans quel ordre sont les coups ? Le mécanisme
    d'attention, seul, n'en a aucune idée : il traite ses entrées comme un sac
    d'éléments. Il faut donc lui injecter la position.

    L'approche historique ajoutait un vecteur de position à chaque token. RoPE
    fait autrement : au lieu d'ajouter une information de position, il **fait
    tourner** les vecteurs requête et clé d'un angle proportionnel à leur
    position. Les dimensions sont prises deux par deux et chaque paire subit une
    rotation dans son plan.

    L'élégance de la chose : quand on calcule ensuite le produit scalaire entre
    une requête en position m et une clé en position n, le résultat ne dépend
    plus que de la différence m - n. L'attention devient donc naturellement
    sensible aux distances relatives, « le coup d'il y a trois demi-coups »
    plutôt que « le coup numéro 47 ». Pour des parties d'échecs c'est
    exactement ce qu'on veut : un motif tactique a la même signification qu'il
    survienne au coup 10 ou au coup 40.
    """
    # Chaque paire de dimensions tourne à une vitesse différente : les
    # premières tournent vite (elles encodent les positions proches), les
    # dernières tournent lentement (elles encodent la position globale).
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                                / head_dim))
    positions = torch.arange(block_size, device=device).float()
    angles = torch.outer(positions, inv_freq)          # (T, head_dim/2)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Applique la rotation à un tenseur (B, n_head, T, head_dim)."""
    # On sépare les dimensions en deux moitiés, qui jouent le rôle des parties
    # réelle et imaginaire d'un nombre complexe.
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Attention multi-tête causale.

    « Causale » signifie que le coup en position t ne peut regarder que les
    coups 0 à t. C'est ce qui rend la tâche non triviale : sans ce masque, le
    modèle verrait le coup qu'on lui demande de prédire.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # Les trois projections q, k, v sont fusionnées en une seule matrice :
        # un gros produit matriciel est plus efficace que trois petits.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # PyTorch choisit tout seul l'implémentation la plus rapide disponible
        # (FlashAttention sur ces cartes), qui ne matérialise jamais la matrice
        # d'attention T x T en mémoire.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Le bloc de calcul « par position » du Transformer.

    Un MLP classique fait : projection vers le haut, non-linéarité, projection
    vers le bas. SwiGLU ajoute un mécanisme de porte : on calcule deux
    projections vers le haut, on passe la première dans une fonction SiLU, et
    on **multiplie** les deux résultats terme à terme. La seconde projection
    agit donc comme un robinet qui laisse passer plus ou moins chaque
    dimension de la première.

    L'intuition : plutôt que d'appliquer la même transformation à tout ce qui
    passe, le réseau apprend à décider, dimension par dimension et en fonction
    de l'entrée, ce qui mérite d'être transmis à la couche suivante.

    Ça coûte une troisième matrice là où un MLP classique en a deux. C'est
    pourquoi on réduit la dimension cachée d'un facteur 2/3 par rapport aux 4x
    habituels : 4 x 2/3 = 8/3, soit ici 512 x 8/3 = 1365, arrondi à 1408 pour
    tomber sur un multiple de 64 (les cœurs tensoriels des GPU aiment ça).
    Le nombre de paramètres reste ainsi comparable à celui d'un MLP classique.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Bloc
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Un étage du Transformer, en pre-norm.

    « Pre-norm » veut dire qu'on normalise **avant** chaque sous-couche, et non
    après. La différence paraît cosmétique, elle ne l'est pas : en pre-norm, le
    chemin qui traverse les connexions résiduelles ne rencontre aucune
    normalisation. Le gradient peut donc remonter de la sortie jusqu'à la
    première couche sans être altéré, ce qui rend les réseaux profonds
    entraînables sans phase de chauffe acrobatique. C'est ce qui a permis
    d'empiler des dizaines de couches de façon fiable.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ---------------------------------------------------------------------------
# Le modèle
# ---------------------------------------------------------------------------

class ChessGPT(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_final = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # La matrice d'embedding et celle de sortie sont la même. C'est
            # défendable : les deux encodent « quel coup ressemble à quel
            # coup ». Ça économise un million de paramètres et ça régularise.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Les projections de sortie de chaque bloc sont initialisées plus
        # petit, en 1/sqrt(2 * n_layer). Sans ça, la variance du flux résiduel
        # croît avec la profondeur : chaque bloc ajoute sa contribution, et
        # après seize couches le signal a explosé.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

        cos, sin = build_rope_cache(cfg.block_size,
                                    cfg.n_embd // cfg.n_head,
                                    cfg.rope_theta,
                                    device="cpu", dtype=torch.float32)
        # register_buffer : ces tenseurs suivent le modèle sur le GPU et dans
        # les checkpoints, mais ne sont pas des paramètres à entraîner.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"séquence de {T} tokens, contexte limité à {self.cfg.block_size}"

        x = self.tok_emb(idx)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    # -- comptages, utiles pour le MFU et pour la documentation --------------

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            # Si les poids sont liés, lm_head partage la même matrice et a
            # déjà été retiré ; sinon il faut le retirer aussi.
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Estime le coût en FLOP d'un token, passe avant + arrière.

        Méthode dite « de Chinchilla » : on compte 6 FLOP par paramètre et par
        token (2 pour la passe avant, 4 pour la passe arrière), auxquels on
        ajoute le coût de l'attention, qui lui dépend de la longueur du
        contexte et n'est donc pas capté par le comptage de paramètres.

        Ce chiffre sert à calculer le MFU (Model FLOPs Utilization) : la part
        de la puissance brute du GPU qu'on exploite réellement. Un MFU de 40 %
        sur ce genre de modèle est un bon score ; en dessous de 20 %, il y a
        un goulot à chercher.
        """
        cfg = self.cfg
        n = self.num_params(non_embedding=True)
        # 6 * N pour les couches denses
        flops = 6 * n
        # Attention : les deux produits QK^T et (attn @ V), passe avant et
        # arrière, soit 12 FLOP par (couche, tête, position vue).
        flops += 12 * cfg.n_layer * cfg.n_head * (cfg.n_embd // cfg.n_head) * cfg.block_size
        return flops

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 allowed_mask_fn=None):
        """Génère des coups un par un.

        `allowed_mask_fn` permet, si on le souhaite, de restreindre les coups
        possibles à chaque étape (c'est ce que fera engine.py pour garantir la
        légalité). Laissé à None, le modèle génère librement : c'est le mode
        utilisé pour mesurer le taux de coups légaux, qui est le chiffre
        central de ce projet.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if allowed_mask_fn is not None:
                mask = allowed_mask_fn(idx)
                logits = logits.masked_fill(~mask, float("-inf"))

            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]],
                                                float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
        return idx


if __name__ == "__main__":
    cfg = ModelConfig()
    model = ChessGPT(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"Configuration : {cfg.n_layer} couches, d={cfg.n_embd}, "
          f"{cfg.n_head} têtes, MLP caché {cfg.mlp_hidden}, "
          f"contexte {cfg.block_size}")
    print(f"  paramètres totaux        : {total:,}")
    print(f"  paramètres non-embedding : {model.num_params():,}")
    print(f"  FLOP par token (fwd+bwd) : {model.flops_per_token():,.0f}")

    # Test de passe avant avec la vraie tâche : prédire le token *suivant*.
    # Attention au piège : évaluer le modèle sur `targets = inputs` (prédire le
    # token courant) donne une loss bien inférieure au hasard dès
    # l'initialisation. Ce n'est pas un bug du modèle mais une propriété des
    # embeddings liés : la connexion résiduelle transporte l'embedding du token
    # jusqu'à la sortie, où lm_head, qui *est* la matrice d'embedding, le
    # reconnaît. Le réseau sait donc recopier son entrée avant tout
    # entraînement. Il ne sait rien prédire pour autant.
    seq = torch.randint(0, cfg.vocab_size, (2, cfg.block_size + 1))
    inputs, targets = seq[:, :-1], seq[:, 1:]
    logits, loss = model(inputs, targets)
    print(f"  test de passe avant      : logits {tuple(logits.shape)}, "
          f"loss {loss.item():.4f}")
    print(f"  loss attendue au hasard  : {math.log(cfg.vocab_size):.4f}")

    _, loss_identity = model(inputs, inputs)
    print(f"  (contrôle) loss si la cible est l'entrée elle-même : "
          f"{loss_identity.item():.4f}")
