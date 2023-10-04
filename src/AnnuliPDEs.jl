module AnnuliPDEs

using AnnuliOrthogonalPolynomials, ClassicalOrthogonalPolynomials, 
    MultivariateOrthogonalPolynomials, LinearAlgebra,
    SemiclassicalOrthogonalPolynomials, DelimitedFiles, LaTeXStrings, PyPlot, 
    BlockArrays, StaticArrays

import AnnuliOrthogonalPolynomials: Laplacian, unweighted, blockcolsupport, ZernikeAnnulusITransform, OneTo, weight
import ForwardDiff: derivative
import MultivariateOrthogonalPolynomials: ZernikeITransform, ModalInterlace, ModalTrav
import ClassicalOrthogonalPolynomials: HalfWeighted, pad

export Laplacian, ModalTrav, derivative, ZernikeAnnulus, HalfWeighted,
    weighted_zernike_modal_solve, zernike_modal_solve, chebyshev_fourier_helmholtz_modal_solve, twoband_fourier_helmholtz_modal_solve,
    plot_solution, collect_errors,
    chebyshev_fourier_zernike_helmholtz_modal_solve, chebyshev_fourier_annulus_helmholtz_modal_solve,
    zernikeannulus_chebyshev_fourier_helmholtz_modal_solve,
    _collect_errors,
    get_rs, get_θs

include("errors.jl")
include("plotting.jl")
include("solver.jl")

get_rs(x) = x.r
get_θs(x) = x.θ

end # module
