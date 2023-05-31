module AnnuliPDEs

using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, 
    MultivariateOrthogonalPolynomials, LinearAlgebra,
    SemiclassicalOrthogonalPolynomials, DelimitedFiles, LaTeXStrings, PyPlot, BlockArrays

import AlgebraicCurveOrthogonalPolynomials: Laplacian, unweighted, blockcolsupport, ZernikeAnnulusITransform, OneTo, weight
import ForwardDiff: derivative
import MultivariateOrthogonalPolynomials: ZernikeITransform, ModalInterlace, ModalTrav
import ClassicalOrthogonalPolynomials: HalfWeighted

export Laplacian, ModalTrav, derivative, ZernikeAnnulus, HalfWeighted,
    helmholtz_modal_solve, chebyshev_fourier_helmholtz_modal_solve, twoband_fourier_helmholtz_modal_solve,
    plot_solution, collect_errors

include("errors.jl")
include("plotting.jl")
include("solver.jl")

end # module
