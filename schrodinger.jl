using LinearAlgebra
using Printf
using SparseArrays

using Plots

module Lanczos

	using LinearAlgebra

	"""
	Return a tridiagonal matrix `T` and orthonormal matrix `V`
	such that `T = V' * A * V`.
	"""
	function lanczos(A, m, v)
		if ndims(A) != 2 || size(A)[1] != size(A)[2]
			error("A must be a square matrix")
		end
		n = size(A, 1)
		if (m > n)
			error("m must be at most size of A")
		end
		α = zeros(m)
		β = zeros(m)
		V = zeros(n, m)

		v = v ./ norm(v)
		V[:,1] = v
		w = A * v
		α[1] = w' * v
		w = w - α[1] * v

		for j in 2:m
			β[j] = norm(w)
			if β[j] != 0
				V[:,j] = w ./ β[j]
			else
				error("β == 0, j = $j")
				v = randn(n)
				V[:,j] = v ./ norm(v)
			end
			w = A * V[:,j]
			α[j] = w' * V[:,j]
			w = w - α[j] * V[:,j] - β[j] * V[:,j-1]
		end
		T = SymTridiagonal(α, β[2:end])
		T, V
	end

	lanczos(A, m) = lanczos(A, m, randn(size(A, 1)))
	lanczos(A) = lanczos(A, round(Int, size(A, 1) / 2))

	"""
	Return the first `m` eigenvectors of matrix `A`.

	This implementation is based on the Wikipedia article on Lanczos method.
	"""
	function eigenvectors(A, args...)
		T, V = lanczos(A, args...)
		t = LinearAlgebra.eigvecs(T)
		V * t
	end
end

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

N = 201
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

with(ratio=1, xlabel="x [nm]", ylabel="y [nm]") do
	global pg = heatmap(x, y, Vg)
	global pb = heatmap(x, y, Vb)
end
# display(pg)
# display(pb)

eg = Lanczos.eigenvectors(Hg, 800, ones(N^2))
eb = Lanczos.eigenvectors(Hb, 800, ones(N^2))
erg = reshape(eg, N, N, :)
erb = reshape(eb, N, N, :)

with(ratio=1, xlabel="x [nm]", ylabel="y [nm]") do
	heatmap(x, y, erg[:,:,1])
end
