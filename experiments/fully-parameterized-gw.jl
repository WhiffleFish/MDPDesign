using POMDPs
using POMDPTools
using POMDPModels
using MDPDesign
using StaticArrays
using MDPDesign.ParameterizedModels
using Plots

gw = FullyParameterizedGridWorld()
θ = fill(0.7, 100)

frozen_mdp = MDPDesign.FrozenParameterizedTabularMDP(MDPDesign.ParameterizedTabularMDP(gw), θ)
mdp_sol = MDPDesign.OptimizedMDPModel(frozen_mdp)
grad = MDPDesign.parameterization_gradient(mdp_sol)

positions = vcat(
    [GWPos(x,y) for (x,y) in zip(3:9, ones(Int,length(3:9)) * 4)],
    [GWPos(x,y) for (x,y) in zip(ones(Int,length(reverse(3:4))) * 10, reverse(3:4))],
    [GWPos(x,y) for (x,y) in zip(reverse(4:10), ones(Int,length(reverse(4:10))) * 2)],
    [GWPos(x,y) for (x,y) in zip(ones(Int,length(2:3)) * 3, 2:3)]
)


max_grad = mapreduce(max, positions) do pos
    maximum(grad[stateindex(gw.mdp, pos)])
end

r_pos = first.(collect(gw.mdp.rewards))
r = last.(collect(gw.mdp.rewards))

anim = @animate for pos ∈ positions
    V_grid = zeros(gw.mdp.size...)
    S = states(gw.mdp)[1:end-1]
    s_idx = stateindex(gw.mdp, pos)
    for i ∈ eachindex(S)
        V_grid[S[i]...] = grad[s_idx][i]
    end
    heatmap(V_grid',xticks=false,yticks=false, clims=(0, 5), cmap=:BuPu, legend=false, size=(700,700), primary=true)
    scatter!(first.(r_pos), last.(r_pos), marker_z=r, cmap=cgrad([:red, :white, :green], [-10, 0, 10]), label="",legend=false, ms=20, primary=false)
    scatter!([first(pos)], [last(pos)], c=:white, ms=20, primary=false)
end

gif(anim, "fully-parameterized.gif", fps = 3)


pos2 = vcat(
    [GWPos(x,y) for (x,y) in zip(ones(9), 1:9)],
    [GWPos(x,y) for (x,y) in zip(ones(9), reverse(2:10))],
)

anim2 = @animate for pos ∈ pos2
    V_grid = zeros(gw.mdp.size...)
    S = states(gw.mdp)[1:end-1]
    s_idx = stateindex(gw.mdp, pos)
    for i ∈ eachindex(S)
        V_grid[S[i]...] = grad[i][s_idx]
    end
    heatmap(V_grid',xticks=false, yticks=false, legend=false, size=(700,700), primary=true)
    # scatter!(first.(r_pos), last.(r_pos), marker_z=r, cmap=cgrad([:red, :white, :green], [-10, 0, 10]), label="",legend=false, ms=20, primary=false)
    scatter!([first(pos)], [last(pos)], c=:white, ms=20, primary=false)
end
gif(anim2, "fully-parameterized-lhs-paths.gif", fps = 3)
