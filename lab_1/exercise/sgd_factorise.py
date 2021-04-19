from typing import Tuple
import torch

def sgd_factorise(A: torch.Tensor, rank: int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    U = torch.rand((m, rank))
    V = torch.rand((n, rank)) 

    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                e = A[r, c] - (U[r] @ V[c].T)
                U[r] += lr * e * V[c]
                V[c] += lr * e * U[r]

    return U, V

def sgd_factorise_masked(A: torch.Tensor, M: torch.Tensor, rank: int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    U = torch.rand((m, rank))
    V = torch.rand((n, rank)) 

    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                if M[r, c]:
                    e = A[r, c] - (U[r] @ V[c].T)
                    U[r] += lr * e * V[c]
                    V[c] += lr * e * U[r]

    return U, V