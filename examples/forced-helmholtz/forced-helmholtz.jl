using AnnuliPDEs, AnnuliOrthogonalPolynomials, ClassicalOrthogonalPolynomials
using PyPlot, Plots, LaTeXStrings
using DelimitedFiles
"""
This script implements the "Forced Helmholtz equation" example (section 7.1).

We are approximating the solution of
    (Δ + 80²)u = sin(100x)
on the annuli domains with inradii 0.2, 0.5, and 0.8.

The errors are approximated by comparing against a high order reference solution.

"""
ρs = [0.2, 0.5, 0.8]   # inradii
NTF = [1000, 500, 500] # reference truncation degree for Chebyshev-Fourier
NZ = [500, 500, 300]   # reference truncation degree for Zernike annular
k = 80                 # forcing coefficient

# RHS in Cartesian coordinates
rhs_xy(x, y) =  sin(100*x)

# Loop over domains
for it in 1:3
    ρ = ρs[it]

    ###
    # Chebyshev-Fourier series discretisation
    ###
    T,C,F = chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
    r = axes(T,1)
    D = Derivative(r)

    Lw = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂ = (r²*Δ)
    M = C\T # Identity
    R = C \ (r .* C) # mult by r

    # Compute high-p reference solution
    Xf, Ff = chebyshev_fourier_helmholtz_modal_solve((T, F), (Lw, M, R), rhs_xy, NTF[it]-2, k^2, rFs=true) 

    Z = ZernikeAnnulus(ρ,1,1)
    wZ = Weighted(Z)
    # Oversampled grid for comparing errors of the solution
    G = AnnuliOrthogonalPolynomials.grid(ZernikeAnnulus(ρ,1,1), Block(1000))

    rs = get_rs.(G)[:,1]
    θs = get_θs.(G)[1,:]
    # Reference solution of Chebyshev-Fourier evaluated on G
    U = (F[θs,1:size(Xf,2)]*(T[rs,1:size(Xf,1)]*Xf)')'

    # Collect errors of Chebyshev-Fourier discretisation
    errors_TF = []
    # TNs = 5:5:200
    TNs = 5:5:10
    for n in TNs
        X = chebyshev_fourier_helmholtz_modal_solve((T, F), (Lw, M, R), rhs_xy, n-2, k^2) 
        collect_errors((T,F,ρ), X, U, G, errors_TF)
        print("Chebyshev-Fourier: Computed coefficients for n=$n \n")
    end
    errors_TF
    writedlm("errors_TF-rho-$ρ.log", errors_TF)

    ###
    # Plotting of solution & RHS
    ###
    # G2 = AnnuliOrthogonalPolynomials.grid(Z, Block(500))
    # rs = get_rs.(G2)[:,1]
    # θs = get_θs.(G2)[1,:]
    # U2 = (F[θs,1:size(Xf,2)]*(T[rs,1:size(Xf,1)]*Xf)')'
    # F2 = (F[θs,1:size(Ff,2)]*(T[rs,1:size(Ff,1)]*Ff)')'
    # plot_solution([Matrix(U2)], [G2], inner_radial=ρ)
    # # PyPlot.savefig("forced-helmholtz-rho-$ρ.pdf")
    # PyPlot.savefig("forced-helmholtz-rho-$ρ.png", dpi=700)
    # plot_solution([Matrix(F2)], [G2], ttl=latexstring(L"$f(x,y)$"), inner_radial=ρ)
    # # PyPlot.savefig("forced-helmholtz-rho-8e-1-f.pdf")
    # PyPlot.savefig("forced-helmholtz-rho-$ρ-f.png", dpi=700)
    # # PyPlot.savefig("forced-helmholtz.pdf")

    ###
    # Zernike annular discretisation
    ###
    Δ = Z \ (Laplacian(axes(Z,1)) * wZ); # Laplacian
    L = Z \ wZ;                          # Identity lowering
    Δs, Ls = Δ.ops, L.ops;
    Δs[NZ[it]]; # Preallocating speeds things up
    Ls[NZ[it]];

    xy = axes(Z,1); x,y = first.(xy),last.(xy)

    # Build high-p reference solution
    fz = Z[:, Block.(1:NZ[it])] \ rhs_xy.(x, y)
    uz = weighted_zernike_modal_solve(fz, NZ[it], Δs, Ls, k^2)
    # Uz is the high-p reference solution evaluated on G.
    _, Uz = collect_errors(wZ, uz, U, G, rvals=true)

    # Collect error for Zernike
    # Ns = 5:5:(NZ[it]-100)
    Ns = 5:5:10
    errors_wZ = []
    for n in reverse(Ns)
        fz = Z[:, Block.(1:n)] \ rhs_xy.(x, y)
        uz = weighted_zernike_modal_solve(fz, n, Δs, Ls, k^2)
        collect_errors(wZ, uz, Uz, G, errors_wZ)
        print("Zernike annular: Computed coefficients for n=$n \n")
    end
    errors_wZ = reverse(errors_wZ)
    writedlm("errors_wZ-rho-$ρ.log", errors_wZ)


    errors_TF = readdlm("errors_TF-rho-$ρ.log")
    errors_wZ = readdlm("errors_wZ-rho-$ρ.log")

    # Plots the convergence of the methods
    ns = [sum(1:n) for n in Ns]
    Plots.plot(ns, errors_wZ,
    label=L"$\mathrm{Zernike \,\, annular}$",
        linewidth=2,
        markershape=:circle,
        markersize=5
    )

    tfns = [(2n-1)*(n+1) for n in TNs]
    Plots.plot!(tfns, errors_TF[1:length(tfns)],
        label=L"$\mathrm{Chebyshev} \otimes \mathrm{Fourier}$",
        linewidth=2,
        markershape=:dtriangle,
        markersize=5,
        # legend=:bottomleft,
        ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
        xlabel=L"$\# \mathrm{Basis \; functions}$",
        yscale=:log10,
        xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
        gridlinewidth = 2
        # yticks=[1e-15,1e-10,1e-5,1e-3,1e-1], # 1e-5
        # ylim=[1e-6,5e-1] # 1e-5
    )
    Plots.savefig("high-frequency-convergence-rho-$ρ.pdf")
end