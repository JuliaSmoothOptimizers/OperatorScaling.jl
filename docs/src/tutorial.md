# Equilibration algorithm

The equilibration algorithm scales a matrix so that its rows and columns have an infinite norm of 1.

```@example equilibration
using LinearAlgebra, SparseArrays
using OperatorScaling
T = Float64
ϵ = 1.0e-4 # tolerance
m, n = 7, 5
A = sprand(T, m, n, 0.6)
A_scaled, D1, D2 = equilibrate(A, ϵ = ϵ)
norm(D1 * A * D2 - A_scaled) ≤ sqrt(eps(T)) * norm(A)
```

Display of the input matrix `A`:

```@example equilibration
A
```

Display of the scaled matrix `A_scaled`:
```@example equilibration
A_scaled
```

The in-place version uses storage diagonal matrices, and updates `A`, `D1` and `D2` in-place.

```@example equilibration
D1, R_k = Diagonal(Vector{T}(undef, m)), Diagonal(Vector{T}(undef, m))
D2, C_k = Diagonal(Vector{T}(undef, n)), Diagonal(Vector{T}(undef, n))
A_scaled2 = copy(A)
equilibrate!(A_scaled2, D1, D2, R_k, C_k, ϵ = ϵ)
# A_scaled2, D1 and D2 are now updated
norm(D1 * A * D2 - A_scaled2) ≤ sqrt(eps(T)) * norm(A)
```

This packages also features an implementation for symmetric matrices:

```@example equilibration
A = sprand(Float64, m, m, 0.3)
Q = Symmetric(tril(A + A'), :L)
Q_scaled, D = equilibrate(Q, ϵ = ϵ)
norm(D * Q * D - Q_scaled) ≤ sqrt(eps(T)) * norm(Q)
```

Display of the input matrix `Q`:

```@example equilibration
Q
```

Display of the scaled matrix `Q_scaled`:
```@example equilibration
Q_scaled
```

```@example equilibration
Q_scaled2 = copy(Q)
# D diagonal matrix and storage diagonal matrix (same size as Q)
D, C_k = Diagonal(Vector{T}(undef, m)), Diagonal(Vector{T}(undef, m))
equilibrate!(Q_scaled2, D, C_k, ϵ = ϵ)
norm(D * Q * D - Q_scaled2) ≤ sqrt(eps(T)) * norm(Q)
```
