"""
Helper functions for plotting the functions on the annulus
"""

function plot_solution(Z::Vector{MultivariateOrthogonalPolynomial{2,T}}, u::Tuple{ModalTrav{T, Matrix{T}}, ModalTrav{T, Matrix{T}}}) where T

    # Extract parameters
    (a, b) = Z[2].a, Z[2].b
    if Z[1] isa Weighted
        (α, β, ρ) = Z[1].P.a, Z[1].P.b, Z[1].P.ρ 
        w = weight(Z[1])
    else
        (α, β, ρ) = Z[1].a, Z[1].b, Z[1].ρ 
    end
    
    # Synthesis operators
    N = 2*size((ModalTrav(u[1]).matrix),1)-1
    F = [
        ZernikeAnnulusITransform{T}(N, α, β, 0, ρ)
        ZernikeITransform{T}(N, a, b)
    ]
    
    # Synthesis
    vals = [F[1] * u[1]] # transform to grid
    append!(vals, [F[2]*u[2]])
    # Synthesis grid
    g = [
        AlgebraicCurveOrthogonalPolynomials.grid(Z[1], Block(N)), 
        AlgebraicCurveOrthogonalPolynomials.grid(Z[2], Block(N))
    ]

    # ammend if weighted
    if Z[1] isa Weighted
        vals[1] = w[g[1]].*vals[1]
    end

    # Use PyPlot to produce nice plots
    p = g -> [g.r, g.θ]
    rθ = [map(p, g[1])]
    append!(rθ, [map(p, g[2])])
    r = [first.(rθ[1])[:,1], ρ*first.(rθ[2])[:,1]]
    θ = [last.(rθ[1])[1,:], last.(rθ[2])[1,:]]

    θ = [[θ[1]; 2π], [θ[2]; 2π]]
    vals[1] = hcat(vals[1], vals[1][:,1])
    vals[2] = hcat(vals[2], vals[2][:,1])
    PyPlot.rc("font", family="serif", size=14)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
    rcParams["text.usetex"] = true
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")

    pc = pcolormesh(θ[1], vcat(r[1],r[2]), vcat(vals[1],vals[2]), cmap="bwr", shading="gouraud")
    
    cbar = plt.colorbar(pc, pad=0.2)
    cbar.set_label(latexstring(L"$u(x,y)$"))
    display(gcf())
end