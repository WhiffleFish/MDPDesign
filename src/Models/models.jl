module ParameterizedModels

using POMDPs
using POMDPTools
using POMDPModels
using StaticArrays
using ..MDPDesign

include("gridworld.jl")
export UniformlyParameterizedGridWorld, FullyParameterizedGridWorld


end
