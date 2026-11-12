"""T-039 (2026-10-19, CPU, ~minutes). Does the 16-slot C label alone determine (alpha, beta, pi)?  Synthetic: C_s = alpha (x) S[pi(s)] xor beta."""
import sys
import numpy as np
sys.path.insert(0, "/home/lantian/ASCAD")
from MultiResNet.decoder_loss import GF_MUL

rng = np.random.default_rng(0)
INV = np.zeros(256, dtype=np.int64)
for a in range(1, 256):
    INV[a] = int(np.where(GF_MUL[a] == 1)[0][0])
N = 3000
counts, pi_ok, with_dup = [], 0, 0
for _ in range(N):
    S = rng.integers(0, 256, 16)                      # S_b = Sbox(p_b xor k_b): uniform for random plaintext
    pi = rng.permutation(16)
    a = int(rng.integers(1, 256)); b = int(rng.integers(0, 256))
    C = GF_MUL[a, S[pi]] ^ b                          # C_slot = alpha (x) S[pi(slot)] xor beta
    sset = np.zeros(256, bool); sset[S] = True
    sols = []
    for a2 in range(1, 256):
        t = INV[a2]
        for b2 in range(256):
            if sset[GF_MUL[t, C ^ b2]].all():
                sols.append((a2, b2))
    counts.append(len(sols))
    with_dup += len(set(S.tolist())) < 16
print("instances", N, "| solutions for (alpha,beta) given C and the S set: min/max", min(counts), max(counts),
      "| exactly one:", sum(c == 1 for c in counts), "| instances with repeated S values:", with_dup)
