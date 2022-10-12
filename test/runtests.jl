using LinearAlgebra, SparseArrays
using Scaling
using Test

@testset "Equilibration" begin
  T = Float64
  ϵ = 1.0e-2
  m, n = 15, 20
   # no empty rows/cols to be able to test infinity row/col norm
  A1 = sprand(T, m, n, 0.2) + 0.5 * I
  A1[1, m+1:n] .+= 0.5
  A2 = rand(T, m, n)
  for A in [A1, A2]
    A_scaled, D1, D2 = equilibrate(A, ϵ = ϵ)
    @test all(abs.(maximum(A_scaled; dims = 1) .- one(T)) .≤ ϵ)
    @test all(abs.(maximum(A_scaled; dims = 2) .- one(T)) .≤ ϵ)
    @test norm(D1 * A * D2 - A_scaled) ≤ sqrt(eps(T)) * norm(A)

    # test in-place
    D1, R_k = Diagonal(Vector{T}(undef, m)), Diagonal(Vector{T}(undef, m))
    D2, C_k = Diagonal(Vector{T}(undef, n)), Diagonal(Vector{T}(undef, n))
    A_scaled2 = copy(A)
    equilibrate!(A_scaled2, D1, D2, R_k, C_k, ϵ = ϵ)
    @test all(abs.(maximum(A_scaled2; dims = 1) .- one(T)) .≤ ϵ)
    @test all(abs.(maximum(A_scaled2; dims = 2) .- one(T)) .≤ ϵ)
    @test norm(D1 * A * D2 - A_scaled2) ≤ sqrt(eps(T)) * norm(A)

    # test with tranpose
    AT = SparseMatrixCSC(A')
    AT_scaled, D1, D2 = equilibrate(AT, A_transposed = true, ϵ = ϵ)
    @test all(abs.(maximum(AT_scaled; dims = 1) .- one(T)) .≤ ϵ)
    @test all(abs.(maximum(AT_scaled; dims = 2) .- one(T)) .≤ ϵ)
    @test norm(D2 * AT * D1 - AT_scaled) ≤ sqrt(eps(T)) * norm(A)
    @test norm(AT_scaled - A_scaled') ≤ sqrt(eps(T)) * norm(A)

    # test symmetric
    Q = A * A'
    Q_scaled, D = equilibrate(Symmetric(Q, :L), ϵ = ϵ)
    @test all(abs.(maximum(Q_scaled; dims = 1) .- one(T)) .≤ ϵ)
    @test all(abs.(maximum(Q_scaled; dims = 2) .- one(T)) .≤ ϵ)
    @test norm(D * Q * D - Q_scaled) ≤ sqrt(eps(T)) * norm(Q)

    # test symmetric lower triangle
    Q2 = tril(Q)
    Q_scaled2, D = equilibrate(Symmetric(Q2, :L), ϵ = ϵ)
    @test all(abs.(maximum(Q_scaled2; dims = 1) .- one(T)) .≤ ϵ)
    @test all(abs.(maximum(Q_scaled2; dims = 2) .- one(T)) .≤ ϵ)
    @test norm(D * Q * D - Q_scaled2) ≤ sqrt(eps(T)) * norm(Q)
    @test norm(Q_scaled2 - Q_scaled) ≤ sqrt(eps(T)) * norm(Q)
  end
end
