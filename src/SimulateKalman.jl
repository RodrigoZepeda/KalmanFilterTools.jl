@doc raw"""
SimulateKalman()

Simulate an implementation of the Kalman Filter given by
```math
\begin{aligned}
    x_{k+1} &= A x_k + B u_k + w_k \\
    y_k &= C x_k + D u_k + v_k
\end{aligned}
```

"""
function SimulateKalman(t::Integer; 
    A::Union{Matrix{<:Real},Nothing}  = nothing, 
    B::Union{Matrix{<:Real},Nothing}  = nothing,
    C::Union{Matrix{<:Real},Nothing}  = nothing,
    D::Union{Matrix{<:Real},Nothing}  = nothing,
    u::Union{Matrix{<:Real},Nothing}  = nothing,
    x0::Union{Vector{<:Real},Nothing} = nothing,
    DistributionW::UnivariateDistribution = Normal(0,1),
    DistributionV::UnivariateDistribution = Normal(0,1),
) 

    #Get dimensions
    if !isnothing(A)
        if (dims(A,1) != dims(A,2))
            error("A must be a square matrix")
        else if (!isnothing(n) & n != dims(A,1))
            error("A must be a square matrix of size $n")
        else if (isnothing(n))
            n = dims(A,1);
        end
    end

    if !isnothing(B)
        if (!isnothing(n) & dims(B,1) != n)
            error("B must be a matrix of size $n x m")
        else if (!isnothing(m) & m != dims(B,2))
            error("B must be a matrix of size $n x $m")
        else if (isnothing(m))
            m = dims(B,1);
        else if (isnothing(n))
            n = dims(B,2);
    end

    if !isnothing(C)
        if (!isnothing(n) & dims(C,2) != n)
            error("C must be a matrix of size q x $n")
        else if (!isnothing(q) & q != dims(C,1))
            error("C must be a matrix of size $q x $n")
        else if (isnothing(q))
            q = dims(C,1);
        else if (isnothing(n))
            n = dims(C,2);
    end

    if !isnothing(D)
        if ((!isnothing(m) & dims(D,2) != m) | (!isnothing(q) & dims(D,1) != q))
            error("D must be a matrix of size $q x $m")
        else if (isnothing(m))
            m = dims(D,2)
        else if (isnothing(q))
            q = dims
    end
    

    
end
export SimulateKalman