using LinearAlgebra
using SparseArrays

using Plots

#module Schrodinger

#const global planckr = 1;
#const global planck = 4.135667696E-15;      # [eVs]
const global planckr = 1.054571817E-34;     # [Js]
const global electronmass = 9.1093837E-31;  # [kg]
const global elemcharge = 1.602176634E-19;  # [C]

function potential_gauss(x, y, V0, σ)
	-V0 * exp(-(x^2 + y^2) / 2σ^2)
end

function potential_c6v(x, y, V0, σ, R)
	v = 0
	for n in [1 2 3 4 5 6]
		s, c = sincospi(n / 3)
		v += exp(- ((x - R * c)^2 + (y - R * s)^2) / 2σ^2)
	end
	-V0 * v
end

#function diff2(indices::Vector{CartesianIndex})
#	D = zeros(length(indices), length(indices))
#	#for ind in indices
#end

function diff2(indices)
	n = maximum(indices)::Int
	ni = size(indices, 1)
	nj = size(indices, 2)
	D = spzeros(n, n)
	for i in 1:ni, j in 1:nj
		c = indices[i,j]
		D[c,c] -= 4
		if i > 1
			D[indices[i-1,j],c] += 1
		end
		if i < ni
			D[indices[i+1,j],c] += 1
		end
		if j > 1
			D[indices[i,j-1],c] += 1
		end
		if j < nj
			D[indices[i,j+1],c] += 1
		end
	end
	D
end

function hamiltonian(x, y, potential::Function; mass = 1, Δ = 1)
	D = diff2(LinearIndices((length(x), length(y))))
	T = (-planckr^2 / 2mass) * D / Δ^2
	V = potential.(x', y)
	T, V
end

V0 = 2   # [eV]
σ = 2    # [nm]
potential_gauss(x, y) = potential_gauss(x, y, V0, σ)
V0 = 2   # [eV]
σ = 0.8  # [nm]
R = 2    # [nm]
potential_c6v(x, y) = potential_c6v(x, y, V0, σ, R)

N = 41
L = 10   # [nm]
x = LinRange(-L/2, L/2, N)
y = LinRange(-L/2, L/2, N)
Δ = x[2] - x[1]

Tg, Vg = hamiltonian(x, y, potential_gauss, Δ=Δ, mass=electronmass)
Tb, Vb = hamiltonian(x, y, potential_c6v,   Δ=Δ, mass=electronmass)

(Tg, Tb) = (Tg, Tb) .* (1E18 / elemcharge)

Hg = Tg + spdiagm(Vg[:])
Hb = Tb + spdiagm(Vb[:])

default(c=cgrad(:viridis, rev=true))

pg, pb
with(ratio=1, xlabel="x [nm]", ylabel="y [nm]") do
	global pg = heatmap(x, y, Vg)
	global pb = heatmap(x, y, Vb)
end

display(pg)
display(pb)

#end
