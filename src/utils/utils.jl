return_one_if_zero(val::T) where {T} = (val == zero(T)) ? one(T) : val

# row / col norms of A in v
function get_norm_rc!(v::AbstractVector{T}, A, ax) where {T}
  v .= zero(T)
  if ax == :row
    maximum!(abs, v, A)
  elseif ax == :col
    maximum!(abs, v', A)
  end
  v .= return_one_if_zero.(sqrt.(v))
end

# updates d1, d2, and A to D1 * A * D2
function mul_D1_A_D2!(A, d1, d2, R, C, A_transposed)
  ldiv!(R, A)
  rdiv!(A, C)
  if A_transposed
    d1 ./= C.diag
    d2 ./= R.diag
  else
    d1 ./= R.diag
    d2 ./= C.diag
  end
end

# updates d and Q to D * Q * D
function mul_Q_D!(Q, d, C)
  ldiv!(C, Q)
  rdiv!(Q, C)
  d ./= C.diag
end

include("csc_utils.jl")
