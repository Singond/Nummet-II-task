using LinearAlgebra
using SparseArrays

using Plots

module Lanczos

	using LinearAlgebra

	"""
	Return a tridiagonal matrix `T` and orthonormal matrix `V`
	such that `T = V' * A * V`.
	"""
	function lanczos_wiki(A, m, v)
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
				v = rand(n)
				V[:,j] = v ./ norm(v)
			end
			w = A * V[:,j]
			α[j] = w' * V[:,j]
			w = w - α[j] * V[:,j] - β[j] * V[:,j-1]
		end
		T = SymTridiagonal(α, β[2:end])
		T, V
	end

	lanczos_wiki(A, m) = lanczos_wiki(A, m, randn(size(A, 1)))
	lanczos_wiki(A) = lanczos_wiki(A, round(Int, size(A, 1) / 2))

	"""
	Return the first `m` eigenvectors of matrix `A`.

	This implementation is based on the Wikipedia article on Lanczos method.
	"""
	function eigvecs_wiki(A, args...)
		T, V = lanczos_wiki(A, args...)
		t = LinearAlgebra.eigvecs(T)
		V * t
	end

	"""
	Return the eigenvectors of matrix `A`.

	This is implementation is based on private lecture notes
	and other sources.
	"""
	function eigvecs_custom(A, q)
		n = size(A, 1)
		α = zeros(n+1)
		β = zeros(n)

		q = q ./ norm(q)
		α[1] = q' * A * q
		r = A * q - α[1] * q
		β[1] = norm(r)
		qprev = q
		q = r ./ β[1]

		for j in 2:n
			α[j] = q' * A * q
			r = A * q - α[j] * q - β[j-1] * qprev
			β[j] = norm(r)
			qprev = q
			q = r ./ β[j]
		end
		α[end] = q' * A * q
		α,β
	end

	"""
	Return the eigenvectors of matrix `A`.

	This is a draft implementation based on the Wikipedia article.
	"""
	function eigvecs_old(A)
		if ndims(A) != 2 || size(A)[1] != size(A)[2]
			error("A must be a square matrix")
		end
		n = size(A)[1]
		v = zeros(n)
		α = zeros(n)
		β = zeros(n)
		β[1] = 1
		j = 1
		w = zeros(n)
		while β[j] != 0
			if j != 1
				t = w
				w = v ./ β[j]
				v = -β[j] * t
			end
			v = A * w + v
			j += 1
			α[j] = w' * v
			v -= α[j] * w
			β[j] = norm(v)
		end
		α, β
	end
end

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

display(pg)
display(pb)

# α,β = eigvecs_custom(rand(4,4), rand(4))
# α,β = eigvecs_custom(Hg, Hg[:,1])
# α,β = eigvecs_old(Hg)

# T = SymTridiagonal(α[2:end], β[2:end])
# eg = eigvecs_wiki(T);
# eg = eigvecs_wiki(Array(Hg))

eg = Lanczos.eigvecs_wiki(Hg, 800)
heatmap(reshape(eg[:,1], N, N))
