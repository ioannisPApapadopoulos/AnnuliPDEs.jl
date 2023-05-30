module AnnuliPDEs

using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, 
    MultivariateOrthogonalPolynomials, LinearAlgebra,
    SemiclassicalOrthogonalPolynomials, DelimitedFiles, LaTeXStrings, PyPlot, BlockArrays

import AlgebraicCurveOrthogonalPolynomials: Laplacian, unweighted, blockcolsupport
import AlgebraicCurveOrthogonalPolynomials: ZernikeAnnulusITransform, OneTo, weight
import ForwardDiff: derivative
import MultivariateOrthogonalPolynomials: ZernikeITransform, ModalInterlace, ModalTrav

export Laplacian, ModalTrav, derivative,
    modal_solve, chebyshev_fourier_modal_solve, plot_solution, collect_errors

include("errors.jl")
include("plotting.jl")
include("solver.jl")

end # module
