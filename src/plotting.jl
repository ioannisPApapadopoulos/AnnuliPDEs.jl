"""
Helper functions for plotting the functions on the annulus
"""

# Plotting routine for a one-cell weighted Zernike annular basis
function plot_solution(Z::Weighted{T, ZernikeAnnulus{T}}, u::PseudoBlockVector, ua=[]; ttl=latexstring(L"$u(x,y)$"), N::Int=0, inner_radial::T=0.0) where T
    w = weight(Z)
    (a, b, ρ) = unweighted(Z).a, unweighted(Z).b, unweighted(Z).ρ

    if N == 0
        N = 2*size((ModalTrav(u).matrix),1)-1
    else
        u = pad(u, axes(Z,2))[Block.(1:N)]
    end

    g = AnnuliOrthogonalPolynomials.grid(Z, Block(N))

    F = ZernikeAnnulusITransform{T}(N, a, b, 0, ρ) # 0 should be Z.c if implemented.
    vals = F * u # transform to grid

    g = AnnuliOrthogonalPolynomials.grid(Z, Block(N))
    if Z isa Weighted
        vals = w[g].*vals
    end

    p = g -> [g.r, g.θ]
    rθ = map(p, g)
    r = first.(rθ)[:,1]
    θ = last.(rθ)[1,:]

    if ua != []
        R = first.(rθ)
        Θ = last.(rθ)
        U = map(ua, R, Θ)
        vals = vals - U
    end

    θ = [θ; 2π]
    vals = hcat(vals, vals[:,1])

    # pl = Plots.heatmap(θ, r[end:-1:1], vals[end:-1:1,:], proj=:polar, limits=(0,1))

    PyPlot.rc("font", family="serif", size=14)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
    rcParams["text.usetex"] = true
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")

    if inner_radial > 0.0
        ax.set_ylim(inner_radial,1)
        ax.set_rorigin(0)
        tick_inner_radial = isodd(10*inner_radial) ? inner_radial+0.1 : inner_radial
        ax.set_rticks(tick_inner_radial:0.2:1)
        y_tick_labels = tick_inner_radial:0.2:1
        ax.set_yticklabels(y_tick_labels)
    end

    pc = pcolormesh(θ, r, vals, cmap="bwr", shading="gouraud")
    cbar = plt.colorbar(pc, pad=0.2)
    cbar.set_label(ttl)
    display(gcf())
end

# Plotting routine for a two-cell Zernike + Zernike annular basis
function plot_solution(Z::AbstractArray{<:MultivariateOrthogonalPolynomial{2,<:T}}, u::Tuple{ModalTrav{T, Matrix{T}}, ModalTrav{T, Matrix{T}}}; inner_radial::T=0.0, ttl=latexstring(L"$u(x,y)$"), N::Int=0) where T

    if N == 0
        N = 2*size((ModalTrav(u[1]).matrix),1)-1
    else
        u = [pad(u[i], axes(Z,2))[Block.(1:N)] for i in 1:2]
    end

    if Z[1] isa ZernikeAnnulus && Z[2] isa Zernike

        # Extract parameters
        (α, β, ρ) = Z[1].a, Z[1].b, Z[1].ρ
        (a, b) = Z[2].a, Z[2].b

        # Synthesis operators
        F = [
            ZernikeAnnulusITransform{T}(N, α, β, 0, ρ)
            ZernikeITransform{T}(N, a, b)
        ]
    elseif Z[1] isa ZernikeAnnulus && Z[2] isa ZernikeAnnulus
            # Extract parameters
            (α, β, ρ) = Z[1].a, Z[1].b, Z[1].ρ
            (a, b, c) = Z[2].a, Z[2].b, Z[2].ρ

            # Synthesis operators
            F = [
                ZernikeAnnulusITransform{T}(N, α, β, 0, ρ)
                ZernikeAnnulusITransform{T}(N, a, b, 0, c)
            ]
    else
        error("Z is not a list of Zernike annular + Zernike (annular).")
    end
    
    # Synthesis
    vals = [F[i] * u[i] for i in 1:2] # transform to grid
    # Synthesis grid
    g = [AnnuliOrthogonalPolynomials.grid(Z[i], Block(N)) for i in 1:2]

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

    if inner_radial > 0.0
        ax.set_ylim(inner_radial,1)
        ax.set_rorigin(0)
        tick_inner_radial = isodd(10*inner_radial) ? inner_radial+0.1 : inner_radial
        ax.set_rticks(tick_inner_radial:0.2:1)
        y_tick_labels = tick_inner_radial:0.2:1
        ax.set_yticklabels(y_tick_labels)
    end

    pc = pcolormesh(θ[1], vcat(r[1],r[2]), vcat(vals[1],vals[2]), cmap="bwr", shading="gouraud")
    
    cbar = plt.colorbar(pc, pad=0.2)
    cbar.set_label(ttl)
    display(gcf())
end

# Plotting routine for a provided vals
function plot_solution(vals::Vector{<:Matrix{T}}, G::Vector{<:Matrix{<:RadialCoordinate{T}}};ρ::T=0.0, ttl=latexstring(L"$u(x,y)$"), inner_radial=0.0) where T

    # Use PyPlot to produce nice plots
    r = [get_rs.(G[i][:,1]) for i in 1:lastindex(G)]
    θ = [get_θs.(G[i][1,:]) for i in 1:lastindex(G)]

    θ = [[θ[i]; 2π] for i in 1:lastindex(G)]
    vals_ = [hcat(vals[i], vals[i][:,1]) for i in 1:lastindex(vals)]

    PyPlot.rc("font", family="serif", size=14)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
    rcParams["text.usetex"] = true
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")

    if inner_radial > 0.0
        ax.set_ylim(inner_radial,1)
        ax.set_rorigin(0)
        tick_inner_radial = isodd(10*inner_radial) ? inner_radial+0.1 : inner_radial
        ax.set_rticks(tick_inner_radial:0.2:1)
        y_tick_labels = tick_inner_radial:0.2:1
        ax.set_yticklabels(y_tick_labels)
    end

        # ax.set_rticks(0.6:0.1:1)  # less radial ticks
    # ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    # # ax.grid(true)

    # y_tick_labels = 0.6:0.1:1
    # ax.set_yticklabels(y_tick_labels)
    # # ind = y_tick_labels.index(0.6)  # find index of value 0

    # gridlines = ax.yaxis.get_gridlines()
    # gridlines[1].set_color("k")
    # gridlines[1].set_linewidth(1.5)

    if ρ ≈ 0.0
        pc = pcolormesh(θ[1], r[1], vals_[1], cmap="bwr", shading="gouraud")
    else
        pc = pcolormesh(θ[1], vcat(r[1],ρ*r[2]), vcat(vals_[1],vals_[2]), cmap="bwr", shading="gouraud")
    end

    cbar = plt.colorbar(pc, pad=0.2)
    cbar.set_label(ttl)
    display(gcf())
end