"""Model Z decoupled cross-attention checks. Run on the training box, not locally.

    python sdm_unips/tests/test_decoupled_context.py

Pure-block tests (no Net, no GLC, no backbone): small dims, eval mode so dropout
is off. Three properties are pinned, each of which the decoupled decoder relies on:

1. forward_cross_self matches an explicit per-query reference, in which query i
   attends to exactly the key set (context U) union {its own token}.
2. No query affects another query: perturbing query j leaves query i unchanged.
3. Queries do depend on the context: perturbing U changes the output.
"""
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit('/sdm_unips/', 1)[0])
from sdm_unips.modules.model.transformer import MultiHeadSelfAttentionBlock


def make_block(dim=32, heads=4, dff=64, seed=0):
    torch.manual_seed(seed)
    blk = MultiHeadSelfAttentionBlock(dim, dim, heads, ln=True,
                                      attention_dropout=0.0, dim_feedforward=dff)
    blk.eval()
    return blk


def reference_one_query(blk, x, y, i):
    """Explicit computation for query i: keys = rows of y, plus x[i]."""
    dV, h = blk.dim_V, blk.num_heads
    d = dV // h
    xq = blk.ln0(x)
    Qf = blk.fc_q(xq)                                   # [1,m,dV]
    keys_in = torch.cat([y[0], x[0, i:i + 1]], dim=0)   # [n+1, dV-in]
    K = blk.fc_k(keys_in)                               # [n+1, dV]
    V = blk.fc_v(keys_in)
    q = Qf[0, i].view(h, d)
    K = K.view(-1, h, d).permute(1, 0, 2)               # [h, n+1, d]
    V = V.view(-1, h, d).permute(1, 0, 2)
    logits = torch.einsum('hd,hjd->hj', q, K) / math.sqrt(d)
    probs = torch.softmax(logits, dim=-1)
    A = torch.einsum('hj,hjd->hd', probs, V).reshape(dV)
    O = Qf[0, i] + A
    O_ = blk.ln1(O)
    O = O + blk.fc_o2(F.gelu(blk.fc_o1(O_)))
    return O


def test_matches_reference():
    blk = make_block()
    torch.manual_seed(1)
    x = torch.randn(1, 6, 32)
    y = torch.randn(1, 5, 32)
    out = blk.forward_cross_self(x, y)
    for i in range(x.shape[1]):
        ref = reference_one_query(blk, x, y[0:1], i)
        assert torch.allclose(out[0, i], ref, atol=1e-5), f'query {i} differs'
    print('ok: matches per-query reference')


def test_no_query_to_query_dependence():
    blk = make_block()
    torch.manual_seed(2)
    x = torch.randn(1, 6, 32)
    y = torch.randn(1, 5, 32)
    base = blk.forward_cross_self(x, y)
    x2 = x.clone()
    x2[0, 3] += 5.0                                     # perturb query 3 only
    pert = blk.forward_cross_self(x2, y)
    keep = [0, 1, 2, 4, 5]
    assert torch.allclose(base[0, keep], pert[0, keep], atol=1e-6), \
        'query i changed when another query j changed'
    assert not torch.allclose(base[0, 3], pert[0, 3]), 'query 3 should change'
    print('ok: queries independent of each other')


def test_queries_depend_on_context():
    blk = make_block()
    torch.manual_seed(3)
    x = torch.randn(1, 6, 32)
    y = torch.randn(1, 5, 32)
    base = blk.forward_cross_self(x, y)
    y2 = y.clone()
    y2[0, 0] += 5.0
    pert = blk.forward_cross_self(x, y2)
    assert not torch.allclose(base, pert), 'output ignores the context'
    print('ok: outputs depend on the context set')


if __name__ == '__main__':
    test_matches_reference()
    test_no_query_to_query_dependence()
    test_queries_depend_on_context()
    print('ALL PASS')
