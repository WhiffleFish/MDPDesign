using POMDPs
using POMDPTools
using POMDPModels
using MDPDesign
using StaticArrays
using MDPDesign.ParameterizedModels
using Plots

gw = UniformlyParameterizedGridWorld(discount=0.90)
θ = [0.0]

frozen_mdp = MDPDesign.FrozenParameterizedTabularMDP(MDPDesign.ParameterizedTabularMDP(gw), θ)
mdp_sol = MDPDesign.OptimizedMDPModel(frozen_mdp)
heatmap(ParameterizedModels.vec2grid(gw.mdp.size,mdp_sol.V)', ticks=false, cmap=cgrad([:red, :white, :green], [-10, 0, 10]), title="LP Value")
savefig("lp-value.pdf")
