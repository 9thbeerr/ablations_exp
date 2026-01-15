from sympy import false
from collections.abc import Iterable
import math
from typing import Any, Callable, Dict, Optional
import torch
import torch.nn as nn
import numpy.typing as npt
import numpy as np
from torch.utils.checkpoint import checkpoint


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weight = nn.Parameter(
            data=nn.init.trunc_normal_(torch.empty(out_dim, in_dim)).to(
                device=device, dtype=dtype
            ),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight.T
        return x


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ) -> None:
        ## Num of Embeddings == Vocab Size
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype

        self.weight = nn.Parameter(
            data=nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim)).to(
                device=device, dtype=dtype
            )
        )

    def forward(self, token_ids: torch.IntTensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device=None, dtype=None
    ) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.eps: float = eps
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weight = nn.Parameter(
            data=nn.init.trunc_normal_(torch.empty(d_model)).to(
                device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x.to(torch.float32)
        rms = torch.square(x) + self.eps
        rms = rms.sum(dim=-1, keepdim=True)
        rms = rms / self.d_model
        rms = torch.sqrt(rms)
        x = (x * self.weight) / rms
        x.to(in_dtype)

        return x


class FeedForwardNet(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device="cpu") -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device)
        self.w2 = Linear(d_ff, d_model, device=device)
        self.w3 = Linear(d_model, d_ff, device=device)

    def forward(self, x: torch.Tensor):
        l1 = self.w1(x)
        silu = torch.sigmoid(l1) * l1
        l3 = self.w3(x)
        glu = silu * l3
        output = self.w2(glu)

        return output


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device="cpu") -> None:
        super().__init__()
        self.theta: float = theta
        self.d_k: float = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 0 to 64 with jump by 2 0,2,4,6,...64
        dimension_pairs = torch.arange(0, d_k, 2).float()
        inv_freq = 1.0 / (self.theta ** (dimension_pairs / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.type_as(self.inv_freq)
        freqs = token_positions[..., None] * self.inv_freq[None, None, :]

        # To compute for longer context length we save this sin/cos with max_seq_len rather than token_positions, so if set max_seq_len 1024 then we can rotate 1024 tokens, does we store them in buffer.

        sin = torch.sin(freqs).to(self.device)
        cos = torch.cos(freqs).to(self.device)

        ## split 64 dims embeddings into 32 even starting from 0, jump by 2 for even and 1 and jump by 2 for odd
        even = x[..., 0::2].to(self.device)
        odd = x[..., 1::2].to(self.device)

        rotated_even = (even * cos) - (odd * sin)
        rotated_odd = (even * sin) + (odd * cos)

        x_rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_entry = torch.max(x, dim=dim, keepdim=True).values
    sum_exps = torch.sum(torch.exp(x - max_entry), keepdim=True, dim=dim)
    output = torch.exp(x - max_entry) / sum_exps

    return output


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    dk = torch.tensor(Q.shape[-1], device=Q.device)
    K = torch.einsum("... ij -> ... ji", K)
    QK = Q @ K
    QK = QK / torch.sqrt(dk)
    mask = mask.to(device=Q.device)
    QK = torch.where(mask, QK, -float("inf"))
    softmax_QK = softmax(QK, dim=-1)
    softmax_QK = softmax_QK.to(QK.device)
    attention = softmax_QK @ V

    return attention


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> torch.Tensor:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = int(d_model // self.num_heads)
        self.d_v = int(d_model // self.num_heads)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        queries_shape = Q.shape[-2]
        keys_shape = K.shape[-2]

        mask = torch.tril(torch.ones(queries_shape, keys_shape)).unsqueeze(0)
        mask = mask.bool()

        attention = scaled_dot_product_attention(Q, K, V, mask)

        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        output = self.o_proj(attention)

        return output


class MHARope(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, rope: nn.Module, device="cpu"
    ) -> torch.Tensor:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.d_k = int(d_model // self.num_heads)
        self.d_v = int(d_model // self.num_heads)

        self.q_proj = Linear(d_model, d_model, device=self.device)
        self.k_proj = Linear(d_model, d_model, device=self.device)
        self.v_proj = Linear(d_model, d_model, device=self.device)
        self.output_proj = Linear(d_model, d_model, device=self.device)

        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Q.to(self.device)
        # K.to(self.device)
        # V.to(self.device)

        queries_shape = Q.shape[-2]
        keys_shape = K.shape[-2]

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(queries_shape, keys_shape)).unsqueeze(0)
        mask = mask.bool()

        attention = scaled_dot_product_attention(Q, K, V, mask)

        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        output = self.output_proj(attention)

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module, device="cpu"
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_model // self.num_heads

        self.attn = MHARope(self.d_model, self.num_heads, rope, device)
        self.ffn = FeedForwardNet(d_model, d_ff, device=device)
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(0, x.shape[-2])

        y = self.ln1(x)
        y = self.attn(y, token_positions)
        x = x + y

        y = self.ln2(x)
        y = self.ffn(y)
        y = x + y

        return y


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device="cpu",
        use_gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)
        self.ln_final = RMSNorm(d_model, device=device)

        dk = d_model // num_heads
        rope = RotaryPositionalEmbedding(rope_theta, dk, context_length, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope, device=device)
                for _ in range(num_layers)
            ]
        )

        self.use_gradient_checkpoint = use_gradient_checkpoint

    def forward(self, x: torch.IntTensor) -> torch.FloatTensor:
        output = self.token_embeddings(x)

        for layer in self.layers:
            if self.use_gradient_checkpoint and self.training:
                output = checkpoint(layer, output, use_reentrant=False)
            else:
                output = layer(output)

        output = self.ln_final(output)

        logits = self.lm_head(output)

        return logits


def calculate_cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # batches = inputs.shape[0]
    # seq_len = inputs.shape[1]
    # vocab_size = inputs.shape[-1]

    inputs = inputs.to(targets.device)

    max_logits = inputs.max(keepdim=True, dim=-1).values
    shifted_logits = inputs - max_logits
    logsumexp = torch.log(torch.exp(shifted_logits).sum(dim=-1, keepdim=True))
    log_probs = shifted_logits - logsumexp

    # cross_entropy = 0
    # for idx, target in enumerate(targets):
    #     loss = -log_probs[idx][target]
    #     cross_entropy = cross_entropy + loss

    target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))

    mean_cross_entropy_loss = -target_log_probs.mean()

    # mean_cross_entropy_loss =  cross_entropy/ targets.shape[0]

    # print("Cross Entrop Losss -->", mean_cross_entropy_loss.item())

    perplexity = torch.exp(mean_cross_entropy_loss)

    # print(f"calculated Perplexity -->", perplexity.item())

    return mean_cross_entropy_loss


def appply_casual_weight_decay(param, update, learning_rate, weight_decay):
    mask = (param * update) >= 0
    update = update + (weight_decay * param * mask)
    return param - learning_rate * update


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Dict[str, Any]],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        if lr < 0:
            raise ValueError("Learning Rate should n't be less than 0", lr)

        defaults = {
            "learning_rate": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay_rate": weight_decay,
            "epsilon": eps,
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            learning_rate = group["learning_rate"]
            beta1, beta2 = group["beta1"], group["beta2"]
            epsilon, weight_decay_rate = group["epsilon"], group["weight_decay_rate"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                time_step = state.get("t", 1)

                first_moment = state.get("first_moment", torch.zeros_like(p.data))
                second_moment = state.get("second_moment", torch.zeros_like(p.data))

                gradients = p.grad.data

                ## Updates

                # first_moment = (beta1 * first_moment) + ((1 - beta1) * gradients)
                # second_moment = (beta2 * second_moment) + ((1 - beta2) * torch.square(gradients))
                first_moment.mul_(beta1).add_(gradients, alpha=(1 - beta1))
                second_moment.mul_(beta2).addcmul_(
                    gradients, gradients, value=(1 - beta2)
                )

                bias_correction1 = 1 - beta1**time_step
                bias_correction2 = 1 - beta2**time_step
                # step_size = learning_rate * ((torch.sqrt(torch.tensor(1 - torch.pow(torch.tensor(beta2), time_step))))/ (1 - torch.pow(torch.tensor(beta1),time_step)))
                denom = (second_moment.sqrt() / bias_correction2**0.5).add_(epsilon)
                step_size = learning_rate / bias_correction1

                # # p.data = p.data - (step_size * (first_moment / torch.sqrt(second_moment + epsilon)))
                # p.data.addcdiv_(first_moment, denom, value=-step_size)

                # casual weight _decay
                update = first_moment / denom  # adam update direction

                # sign-gated weight decay
                mask = (update * p.data) >= 0
                update = update + weight_decay_rate * p.data * mask

                # apply final update
                p.data.add_(update, alpha=-step_size)

                state["t"] = time_step + 1
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment

        return loss


def learning_rate_schedule(
    iter_num: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    learning_rate = lr_min

    if iter_num < warmup_iters:
        learning_rate = (iter_num / warmup_iters) * lr_max
    elif iter_num >= warmup_iters and iter_num <= cosine_cycle_iters:
        learning_rate = lr_min + (
            0.5
            * (
                1
                + math.cos(
                    (iter_num - warmup_iters)
                    / (cosine_cycle_iters - warmup_iters)
                    * math.pi
                )
            )
        ) * (lr_max - lr_min)
    elif iter_num > cosine_cycle_iters:
        learning_rate = lr_min

    return learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> float:
    eps = 1e-06
    ## you take global norm of all parametes and then scale, we used to do per gradients of each parameters.
    total_norm = math.sqrt(
        sum(
            [
                torch.norm(input=param.grad, p=2).pow(2)
                for param in parameters
                if param.grad is not None
            ]
        )
    )

    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                if total_norm > max_l2_norm:
                    scale = max_l2_norm / (total_norm + eps)
                    param.grad.mul_(scale)

    return total_norm


def get_batches(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.tensor, torch.tensor]:
    ## we have dataset which 1D array, we want to create batch of context length starting from anywhere,
    ## lets say random index 5 and + 7 context length then y label would be 6 + 7. so we have
    # 5 training data - > 6 as label which is next token
    # this way we create 32 batch size that each is random start to dataset.

    max_start = len(dataset) - context_length  # -1 so we can shift labels
    indices = np.random.randint(0, max_start, size=batch_size)

    # inputs = np.stack([dataset[i:i + context_length] for i in indices])
    # labels = np.stack([dataset[i + 1:i + 1 + context_length] for i in indices])

    inputs = []
    labels = []
    for i in indices:
        inputs.append(dataset[i : i + context_length])
        labels.append(dataset[i + 1 : i + 1 + context_length])

    x = torch.tensor(inputs, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    return (x, y)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
    config={},
):
    model_state = {
        "weights": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
    }
    torch.save(model_state, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model_state = torch.load(src, weights_only=False)
    model.load_state_dict(model_state["weights"])
    optimizer.load_state_dict(model_state["optimizer_state"])
    iterations = model_state["iteration"]

    return iterations
