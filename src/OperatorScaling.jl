module OperatorScaling

# Write your package code here.

using LinearAlgebra, SparseArrays

export equilibrate, equilibrate!

include("utils/utils.jl")
include("equilibration.jl")

end
