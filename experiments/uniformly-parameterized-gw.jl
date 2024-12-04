using POMDPs
using POMDPTools
using POMDPModels
using MDPDesign
using StaticArrays
using MDPDesign.ParameterizedModels
using Plots

gw = UniformlyParameterizedGridWorld()
θ = [0.7]

frozen_mdp = MDPDesign.FrozenParameterizedTabularMDP(MDPDesign.ParameterizedTabularMDP(gw), θ)
mdp_sol = MDPDesign.OptimizedMDPModel(frozen_mdp)
grad = only(MDPDesign.parameterization_gradient(mdp_sol))

∇V = MDPDesign.ParameterizedModels.vec2grid((10,10), grad)
heatmap(∇V', cmap=:BuPu, xticks=false, yticks=false)

