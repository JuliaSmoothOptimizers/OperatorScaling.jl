function get_norm_rc_CSC!(v::AbstractVector{T}, A_colptr, A_rowval, A_nzval::AbstractVector{T}, n, ax) where {T}
  v .= zero(T)
  for j = 1:n
    @inbounds for i = A_colptr[j]:(A_colptr[j + 1] - 1)
      k = ax == :row ? A_rowval[i] : j
      A_nzi_abs = abs(A_nzval[i])
      if A_nzi_abs > v[k]
        v[k] = A_nzi_abs
      end
    end
  end

  v .= sqrt.(v)
  @inbounds @simd for i = 1:length(v)
    if v[i] == zero(T)
      v[i] = one(T)
    end
  end
end

get_norm_rc!(v::AbstractVector{T}, A::SparseMatrixCSC{T, Int}, ax) where {T} =
  get_norm_rc_CSC!(v, A.colptr, A.rowval, A.nzval, size(A, 2), ax)

function get_norm_rc_CSC_sym!(v::AbstractVector{T}, A_colptr, A_rowval, A_nzval::AbstractVector{T}, n) where {T}
  v .= zero(T)
  for j = 1:n
    @inbounds for i = A_colptr[j]:(A_colptr[j + 1] - 1)
      k = A_rowval[i]
      A_nzi_abs = abs(A_nzval[i])
      if A_nzi_abs > v[k]
        v[k] = A_nzi_abs
      end
      if A_nzi_abs > v[j]
        v[j] = A_nzi_abs
      end
    end
  end

  v .= sqrt.(v)
  @inbounds @simd for i = 1:length(v)
    if v[i] == zero(T)
      v[i] = one(T)
    end
  end
end

get_norm_rc!(v::AbstractVector{T}, A::Symmetric{T, SparseMatrixCSC{T, Int}}, ax) where {T} =
  get_norm_rc_CSC_sym!(v, A.data.colptr, A.data.rowval, A.data.nzval, size(A, 2))

function mul_D1_A_D2_CSC!(A_colptr, A_rowval, A_nzval, d1, d2, r, c, A_transposed)
  for j = 1:length(c)
    @inbounds @simd for i = A_colptr[j]:(A_colptr[j + 1] - 1)
      A_nzval[i] /= r[A_rowval[i]] * c[j]
    end
  end

  if A_transposed
    d1 ./= c
    d2 ./= r
  else
    d1 ./= r
    d2 ./= c
  end
end

mul_D1_A_D2!(A::SparseMatrixCSC, d1, d2, R, C, A_transposed) =
  mul_D1_A_D2_CSC!(A.colptr, A.rowval, A.nzval, d1, d2, R.diag, C.diag, A_transposed)

function mul_Q_D_CSC!(Q_colptr, Q_rowval, Q_nzval, d, c)
  for j = 1:length(d)
    @inbounds @simd for i = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      Q_nzval[i] /= c[Q_rowval[i]] * c[j]
    end
  end
  d ./= c
end

mul_Q_D!(Q::SparseMatrixCSC, d, C) = mul_Q_D_CSC!(Q.colptr, Q.rowval, Q.nzval, d, C.diag)
