module MDPDesign

using POMDPs
using POMDPTools
using JuMP
using HiGHS
using LinearAlgebra
using SparseArrays

include("lp-solve.jl")
export LPSolver

include("parameterized.jl")
export ParameterizedMDPWrapper

include("gradients.jl")
export parameterization_gradient

include("Models/models.jl")
export ParameterizedModels

include("optimization.jl")
export RegularizedMDPOptimizationProblem, value_and_grad

end # module MDPDesign
