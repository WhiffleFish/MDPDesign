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
grad = only(MDPDesign.parameterization_gradient(mdp_sol))

∇V = MDPDesign.ParameterizedModels.vec2grid((10,10), grad)
heatmap(∇V', cmap=:BuPu, xticks=false, yticks=false, size=(800,700))
savefig("uniformly-parameterized-grad.png")

using Compose
img = render(gw.mdp)
savefig(img)
img |> SVG("hello-world4.svg")
