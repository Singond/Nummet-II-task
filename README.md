Solving the Schrödinger equation using Lanczos method
=====================================================
An example implementation and application of the Lanczos diagonalization
method on the eigenvalue problem arising in the study of stationary
solutions of the Schrödinger equation.

This is a semestral project for a course on advanced numerical methods.

<div align="center">
  <img src="img/example.png" width="300" height="300"/>
</div>

Running the code
----------------
The code comes in several mostly equivalent versions:
* `schrodinger.jl` is a plain Julia script which displays one of the
  calculated eigenvectors in the end.
  To run this script, simply launch Julia and execute
  `include("schrodinger.jl")`.
* `schrodinger.nb.jl` is a Pluto-notebook version of the above.
  This is an interactive document which displays intermediate results
  and more than just one eigenvector (solution).
  To run this, install [Pluto](https://plutojl.org/), run it with
  `using Pluto; Pluto.run()` in a Julia session and wait for a document
  to open in your web browser. Navigate to the file and select it.
* `schrodinger_makie.nb.jl` is a notebook using an alternative plotting
  package (`Makie.jl`).
  This version is also available on JuliaHub at
  <https://juliahub.com/ui/Notebooks/Singond/School/schrodinger_makie.nb.jl>.
  It exists because the default plotting package, `Plots.jl`,
  currently has issues when running on JuliaHub.
* `schrodinger_write.jl` is just a helper script used to write the results
  of `schrodinger.jl` into files.
