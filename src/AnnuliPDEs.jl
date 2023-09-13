module AnnuliPDEs

using AnnuliOrthogonalPolynomials, ClassicalOrthogonalPolynomials, 
    MultivariateOrthogonalPolynomials, LinearAlgebra,
    SemiclassicalOrthogonalPolynomials, DelimitedFiles, LaTeXStrings, PyPlot, 
    BlockArrays, StaticArrays, ContinuumArrays

import AnnuliOrthogonalPolynomials: Laplacian, unweighted, blockcolsupport, ZernikeAnnulusITransform, OneTo, weight
import ForwardDiff: derivative
import MultivariateOrthogonalPolynomials: ZernikeITransform, ModalInterlace, ModalTrav
import ClassicalOrthogonalPolynomials: HalfWeighted

export Laplacian, ModalTrav, derivative, ZernikeAnnulus, HalfWeighted,
    helmholtz_modal_solve, chebyshev_fourier_helmholtz_modal_solve, twoband_fourier_helmholtz_modal_solve,
    plot_solution, collect_errors,
    chebyshev_fourier_zernike_helmholtz_modal_solve, zernikeannulus_chebyshev_fourier_helmholtz_modal_solve,
    affine

include("errors.jl")
include("plotting.jl")
include("solver.jl")

end # module
