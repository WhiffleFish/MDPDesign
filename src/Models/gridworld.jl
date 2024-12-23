sigmoid(x, a=0.5, k=1) = a/(1+exp(-k*x)) + a
d_sigmoid(x, a=0.5, k=1) = (a*k*exp(k*x))/(1+exp(k*x))^2

function vec2grid(s::Tuple{Int,Int}, v)
    if length(v) == prod(s) + 1
        v = v[1:end-1]
    end
    V = Matrix{eltype(v)}(undef, s[1], s[2])
    LI = LinearIndices(s)
    for i ∈ eachindex(v)
        V[LI[i]...] = v[i]
    end
    return V
end


function UniformlyParameterizedGridWorld(;kwargs...)
    return ParameterizedMDPWrapper(SimpleGridWorld(;kwargs...), uniform_param_transition, uniform_dparam_transition)
end

function FullyParameterizedGridWorld(;kwargs...)
    return ParameterizedMDPWrapper(SimpleGridWorld(;kwargs...), full_param_transition, full_dparam_transition)
end

function uniform_param_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol, θ)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    end
    θ0 = only(θ)
    θ = sigmoid(θ0)

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = θ # probability of transitioning to the desired cell
        else
            prob = (1.0 - θ)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(convert(SVector, destinations), convert(SVector, probs))
end

function uniform_dparam_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol, θ)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return [SparseCat(SA[GWPos(-1,-1)], SA[0.0])]
    end
    θ0 = only(θ)
    θ = sigmoid(θ0)
    dθ = d_sigmoid(θ0)
    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            dprob = 1.0 # probability of transitioning to the desired cell
        else
            dprob = -1/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += dprob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += dprob
        end
    end

    return [SparseCat(convert(SVector, destinations), convert(SVector, probs) .* dθ)]
end

function full_param_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol, θv::AbstractArray)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    end
    s_idx = stateindex(mdp, s)
    θ = sigmoid(θv[s_idx])

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s
    
    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = θ # probability of transitioning to the desired cell
        else
            prob = (1.0 - θ)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(convert(SVector, destinations), convert(SVector, probs))
end

function full_dparam_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol, θv::AbstractArray)
    ∇T_s = convert(Vector{Union{
        SparseCat{SVector{1, SVector{2, Int64}}, SVector{1, Float64}}, 
        SparseCat{SVector{5, SVector{2, Int64}}, SVector{5, Float64}}
        }},
        fill(
            SparseCat(SA[GWPos(-1,-1)], SA[0.0]), 
            prod(mdp.size)
        )
    )
    if s in mdp.terminate_from || isterminal(mdp, s)
        return ∇T_s
    end
    s_idx = stateindex(mdp, s)
    θ0 = θv[s_idx]
    θ = sigmoid(θ0)
    dθ = d_sigmoid(θ0)

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            dprob = 1.0 # probability of transitioning to the desired cell
        else
            dprob = -1/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += dprob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += dprob
        end
    end

    ∇T_s[s_idx] = SparseCat(convert(SVector, destinations), convert(SVector, probs) .* dθ)

    return ∇T_s
end
