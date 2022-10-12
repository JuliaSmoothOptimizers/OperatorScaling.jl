"""
    equilibrate!(A::AbstractMatrix{T},
                 D1::Diagonal{T, S}, D2::Diagonal{T, S},
                 R_k::Diagonal{T, S}, C_k::Diagonal{T, S};
                 ϵ::T = T(1.0e-2), max_iter::Int = 100,
                 A_transposed::Bool = false) where {T, S <: AbstractVector{T}}

Performs the equilibration algorithm to scale `A` so that its rows and columns have a infinity norm of 1 with a tolerance `ϵ`.
`D1` and `D2` are diagonal scaling factors of sizes `size(A, 1)` and `size(A, 2)` respectively.
Once `A` is scaled, the identity `D1 * A * D2` gives the unscaled matrix.
`R_k` and `C_k` are diagonal matrices of sizes `size(A, 1)` and `size(A, 2)` respectively that should be pre-allocated.
`max_iter` is the maximal number of iterations.

If `A_transposed = true`, then, once `A` is scaled, the identity `D2 * A * D1` gives the unscaled matrix.
When using `A_transposed = true`, `D1` and `D2` should have sizes `size(A, 2)` and `size(A, 1)`.

    equilibrate!(Q::Symmetric{T}, D::Diagonal{T, S}, C_k::Diagonal{T, S};
                 ϵ::T = T(1.0e-2), max_iter::Int = 100) where {T, S <: AbstractVector{T}}

Performs the equilibration algorithm to scale the symmetric matrix `Q` so that its rows and columns have a infinity norm of 1 with a tolerance `ϵ`.
`D` is a diagonal scaling factors of size `size(Q, 1)`.
Once `Q` is scaled, the identity `D * Q * D` gives the unscaled matrix.
`C_k` is a diagonal matrix of size `size(A, 1)` that should be pre-allocated.
`max_iter` is the maximal number of iterations.

# Reference
* D. Ruiz, *A Scaling Algorithm to Equilibrate Both Rows and Columns Norms in Matrices*, RAL-TR-2001-034, 2001.
"""
function equilibrate! end

"""
    A_scaled, D1, D2 = equilibrate(A::AbstractMatrix{T}; A_transposed = false, kwargs...)

Performs the equilibration algorithm on the matrix `A` and returns the scaled matrix matrix `A_scaled` with its diagonal scaling factors `D1` and `D2`.

    Q_scaled, D = equilibrate(Q::Symmetric{T}; kwargs...) where {T}

Performs the equilibration algorithm on the symmetric matrix `Q` and returns the scaled matrix matrix `Q_scaled` with its diagonal scaling factor `D`.

See [`Equilibrate!`](@ref) for the keyword arguments.
"""
function equilibrate end

function equilibrate!(
  A::AbstractMatrix{T},
  D1::Diagonal{T, S},
  D2::Diagonal{T, S},
  R_k::Diagonal{T, S},
  C_k::Diagonal{T, S};
  ϵ::T = T(1.0e-2),
  max_iter::Int = 100,
  A_transposed::Bool = false,
) where {T <: Real, S <: AbstractVector{T}}
  min(size(A)...) == 0 && return
  D1.diag .= one(T)
  D2.diag .= one(T)
  get_norm_rc!(R_k.diag, A, :row)
  get_norm_rc!(C_k.diag, A, :col)
  mul_D1_A_D2!(A, D1.diag, D2.diag, R_k, C_k, A_transposed)
  R_k.diag .= abs.(one(T) .- R_k.diag)
  C_k.diag .= abs.(one(T) .- C_k.diag)
  convergence = maximum(R_k.diag) <= ϵ && maximum(C_k.diag) <= ϵ
  k = 1
  while !convergence && k < max_iter
    get_norm_rc!(R_k.diag, A, :row)
    get_norm_rc!(C_k.diag, A, :col)
    mul_D1_A_D2!(A, D1.diag, D2.diag, R_k, C_k, A_transposed)
    R_k.diag .= abs.(one(T) .- R_k.diag)
    C_k.diag .= abs.(one(T) .- C_k.diag)
    convergence = maximum(R_k.diag) <= ϵ && maximum(C_k.diag) <= ϵ
    k += 1
  end
end

function equilibrate(A::AbstractMatrix{T}; A_transposed = false, kwargs...) where {T}
  R_k = Diagonal(Vector{T}(undef, size(A, 1)))
  C_k = Diagonal(Vector{T}(undef, size(A, 2)))
  if A_transposed
    n, m = size(A)
  else
    m, n = size(A)
  end
  D1 = Diagonal(Vector{T}(undef, m))
  D2 = Diagonal(Vector{T}(undef, n))
  A_scaled = copy(A)
  equilibrate!(A_scaled, D1, D2, R_k, C_k; A_transposed = A_transposed, kwargs...)
  return A_scaled, D1, D2
end

function equilibrate!(
  Q::Symmetric{T},
  D::Diagonal{T, S},
  C_k::Diagonal{T, S};
  ϵ::T = T(1.0e-2),
  max_iter::Int = 100,
) where {T <: Real, S <: AbstractVector{T}}
  size(Q, 1) == 0 && return
  D.diag .= one(T)
  get_norm_rc!(C_k.diag, Q, :col)
  mul_Q_D!(Q.data, D.diag, C_k)
  C_k.diag .= abs.(one(T) .- C_k.diag)
  convergence = maximum(C_k.diag) <= ϵ
  k = 1
  while !convergence && k < max_iter
    get_norm_rc!(C_k.diag, Q, :col)
    mul_Q_D!(Q.data, D.diag, C_k)
    C_k.diag .= abs.(one(T) .- C_k.diag)
    convergence = maximum(C_k.diag) <= ϵ
    k += 1
  end
end

function equilibrate(Q::Symmetric{T}; kwargs...) where {T}
  n = size(Q, 1)
  D = Diagonal(Vector{T}(undef, n))
  C_k = Diagonal(Vector{T}(undef, n))
  Q_scaled = copy(Q)
  equilibrate!(Q_scaled, D, C_k; kwargs...)
  return Q_scaled, D
end
