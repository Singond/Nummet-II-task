### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 36c853f2-1145-11ef-364f-75b6e985de80
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	using LinearAlgebra
	using SparseArrays

	using Plots
end

# ╔═╡ 73286b82-618c-40fb-80c4-f7cd8547b5cd
md"""
# Řešení 2D Schrövnice Lanczosovou diagonalizací
"""

# ╔═╡ ea5bc82e-0283-4bde-8c98-c439edb3c13c
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

# ╔═╡ d0eb243d-e4bb-4a90-87a7-9037682133fc
lanczos_wiki(A, m) = lanczos_wiki(A, m, randn(size(A, 1)))

# ╔═╡ fb023d5b-03ab-4ece-8171-6a92e8ae0335
lanczos_wiki(A) = lanczos_wiki(A, round(Int, size(A, 1) / 2))

# ╔═╡ 754b31c7-4526-4411-9c0f-1c64f2d4d096
"""
Return the first `m` eigenvectors of matrix `A`.

This implementation is based on the Wikipedia article on Lanczos method.
"""
function eigvecs_wiki(A, args...)
	T, V = lanczos_wiki(A, args...)
	t = LinearAlgebra.eigvecs(T)
	V * t
end

# ╔═╡ 9d94d614-ae23-4e58-84ff-0e597877e896
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

# ╔═╡ 2002c9dc-bf92-4706-82c2-f5377ef2a7e8
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

# ╔═╡ aa6ddcbf-83aa-4a77-9f5d-08ff33ffd04a
eigenvectors = eigvecs_wiki

# ╔═╡ 24f01b2e-e881-4e1a-b2bc-c9ad2ec61f5e
md"Planckova konstanta v Js:"

# ╔═╡ dd047f04-ddb2-433a-ac66-15fcdd8e1f96
const global planckr = 1.054571817E-34;

# ╔═╡ fb942c72-f5d1-4f12-a560-cf0a2c85b41d
md"Hmotnost elektronu v kg:"

# ╔═╡ c586a9ac-60b7-40e7-a090-73773d854e18
const global electronmass = 9.1093837E-31;

# ╔═╡ 21a7be76-f249-44f9-83c5-e21cee2c9e49
md"Elementární náboj v C:"

# ╔═╡ 3be03410-64e0-4ac5-843c-9b26657e9f01
const global elemcharge = 1.602176634E-19;

# ╔═╡ 0d61b715-13a3-4ea8-8410-dd86ff8b0c89
md"""
Pomocné funkce:
"""

# ╔═╡ a69d7555-ec14-48ee-a4c4-3272ddaac019
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

# ╔═╡ a5793352-2a91-4cb4-96db-ef21973767f3
function hamiltonian(x, y, potential::Function; mass = 1, Δ = 1)
	D = diff2(LinearIndices((length(x), length(y))))
	T = (-planckr^2 / 2mass) * D / Δ^2
	V = potential.(x', y)
	T, V
end

# ╔═╡ b0efb7d6-cd96-4108-97e8-da8e5d00b65f
md"""
Simulace probíhá na čtvercové oblasti o rozměrech $L \times L$
rozdělené na $N \times N$ uzlů.
"""

# ╔═╡ f6b654c7-0395-422c-b49e-d8a771625903
N = 201

# ╔═╡ f493e015-6831-4344-9dfc-fdc6f04ada72
L = 10   # [nm]

# ╔═╡ 6ae1e759-1f0b-4131-8a0e-d7693875e4e6
md"Pomocné vektory $x$ a $y$ představující prostorové souřadnice:"

# ╔═╡ 336a696f-2545-4f88-928c-1aaed4100988
x = LinRange(-L/2, L/2, N)

# ╔═╡ ff508904-d8f4-406f-a530-877ac212d8d6
y = LinRange(-L/2, L/2, N)

# ╔═╡ 0379f1a2-eaae-4c6d-8324-29b3b0f306ca
md"Diference $x$ (též $y$):"

# ╔═╡ 8b6cf25c-31be-4f0f-b226-7d7137d24ed5
Δ = x[2] - x[1]

# ╔═╡ eb21a633-a15b-4670-89e1-b4beb1d2f9b5
md"Obecné nastavení grafů:"

# ╔═╡ 0edfd386-6b70-4884-aba8-d1964d9cdc97
default(c=cgrad(:viridis, rev=true))

# ╔═╡ d91aec6c-ab70-4538-ae16-3127ecc2a793
md"""
## Gaussovský potenciál

První případ je rotačně symetrický gaussovský potenciál s grupou symetrie O(2)
vyjádřený vztahem:

$$V(x,y) = - V_0 \exp \left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$

kde $V_0 = 2\,\text{eV}$ a $\sigma = 2\,\text{nm}$.
"""

# ╔═╡ 8c51ef55-7e49-424f-85cf-315fa5e92836
function potential_gauss(x, y, V0, σ)
	-V0 * exp(-(x^2 + y^2) / 2σ^2)
end

# ╔═╡ 92694d37-2612-4a36-9809-6fe28811af07
potential_gauss(x, y) = potential_gauss(x, y, 2, 2)

# ╔═╡ da150260-1d36-4a1d-9249-3abab7014ac6
md"Složky hamiltoniánu:"

# ╔═╡ c75116f1-5e8a-438a-a349-047283ac701f
Tg, Vg = hamiltonian(x, y, potential_gauss, Δ=Δ, mass=electronmass)

# ╔═╡ fa2b4e05-e07b-408d-886d-3f375da26f35
with(ratio=1, title="potenciál V_g") do
	heatmap(x, y, Vg, xlabel="x [nm]", ylabel="y [nm]")
end

# ╔═╡ 55452732-7f91-4493-85ef-6dba712f33b0
md"Hamiltonián:"

# ╔═╡ 026934ec-6f67-45f4-bfec-fc44ab6fa913
Hg = Tg * (1E18 / elemcharge) + spdiagm(Vg[:])

# ╔═╡ 648af4da-437f-4207-bb63-200b29ff1243
md"""
Vlastní vektory hamiltoniánu spočtené zvolenou metodou:
"""

# ╔═╡ a3b74a06-c25e-4ecc-bfb1-c669e735809b
eg = eigenvectors(Hg, 800)

# ╔═╡ 3c8b44de-2986-4cbb-aca3-fd88b78dee6b
@bind kg html"k = <input type='number' value='1' min='1' max='41'>"

# ╔═╡ 11a63b16-1f0d-49d4-8a15-0cb726e2627a
with(ratio=1, title="vlastní vektor $(kg)") do
	heatmap(x, y, reshape(eg[:,kg], N, N),
		xlabel="x [nm]", ylabel="y [nm]")
end

# ╔═╡ b14e5668-fbc6-4a8b-b613-c37335abfa77
md"""
## Potenciál molekuly benzenu

Druhý případ je potenciál s grupou symetrie $C_{6v}$, který připomíná
molekulu benzenu:

$$V(x,y) = -V_0 \sum_{n=1}^6 \exp \left[
-\frac{(x - R\cos\frac{n\pi}{3})^2 + (y - R\sin\frac{n\pi}{3})^2}{2\sigma^2}
\right]$$

kde $V_0 = 2\,\text{eV}$, $\sigma = 0{,}8\,\text{nm}$ a $R = 2\,\text{nm}$.
"""

# ╔═╡ b206e6a2-6b00-4863-aecb-833f17e9fc77
function potential_c6v(x, y, V0, σ, R)
	v = 0
	for n in [1 2 3 4 5 6]
		s, c = sincospi(n / 3)
		v += exp(- ((x - R * c)^2 + (y - R * s)^2) / 2σ^2)
	end
	-V0 * v
end

# ╔═╡ 5d3807b6-fdde-4758-8955-77816f2758c5
potential_c6v(x, y) = potential_c6v(x, y, 2, 0.8, 2)

# ╔═╡ 9b00bed5-bf85-45ae-b768-3f4316852f38
md"Složky hamiltoniánu:"

# ╔═╡ 1c99a8a8-2f87-43b9-9316-1724e387ad3d
Tb, Vb = hamiltonian(x, y, potential_c6v, Δ=Δ, mass=electronmass)

# ╔═╡ 2ff59f98-598b-444f-9be6-67b212ee4fe2
with(ratio=1, title="potenciál V_b") do
	heatmap(x, y, Vb, xlabel="x [nm]", ylabel="y [nm]")
end

# ╔═╡ fa3ee0f0-0f8c-4146-ae2c-622050a57999
md"Hamiltonián:"

# ╔═╡ 6579d3bb-bf7e-46c2-ae8b-11627be11b11
Hb = Tb * (1E18 / elemcharge) + spdiagm(Vb[:])

# ╔═╡ e59cfa75-c127-4fde-bea2-af97f2857dcd
md"""
Vlastní vektory hamiltoniánu spočtené zvolenou metodou:
"""

# ╔═╡ 952eeef1-f3bc-48bd-a974-21d4dea587aa
eb = eigenvectors(Hb, 800)

# ╔═╡ 6f7138ab-133a-4e2a-9a56-87754b368450
@bind kb html"<input type='number' value='1' min='1' max='41'>"

# ╔═╡ 16e7014f-d553-4a08-9dec-0e5f33946c3f
with(ratio=1, title="vlastní vektor $(kb)") do
	heatmap(x, y, reshape(eb[:,kb], N, N),
		xlabel="x [nm]", ylabel="y [nm]")
end

# ╔═╡ Cell order:
# ╟─73286b82-618c-40fb-80c4-f7cd8547b5cd
# ╠═36c853f2-1145-11ef-364f-75b6e985de80
# ╠═ea5bc82e-0283-4bde-8c98-c439edb3c13c
# ╠═d0eb243d-e4bb-4a90-87a7-9037682133fc
# ╠═fb023d5b-03ab-4ece-8171-6a92e8ae0335
# ╠═754b31c7-4526-4411-9c0f-1c64f2d4d096
# ╠═9d94d614-ae23-4e58-84ff-0e597877e896
# ╠═2002c9dc-bf92-4706-82c2-f5377ef2a7e8
# ╠═aa6ddcbf-83aa-4a77-9f5d-08ff33ffd04a
# ╟─24f01b2e-e881-4e1a-b2bc-c9ad2ec61f5e
# ╠═dd047f04-ddb2-433a-ac66-15fcdd8e1f96
# ╟─fb942c72-f5d1-4f12-a560-cf0a2c85b41d
# ╠═c586a9ac-60b7-40e7-a090-73773d854e18
# ╟─21a7be76-f249-44f9-83c5-e21cee2c9e49
# ╠═3be03410-64e0-4ac5-843c-9b26657e9f01
# ╟─0d61b715-13a3-4ea8-8410-dd86ff8b0c89
# ╠═a69d7555-ec14-48ee-a4c4-3272ddaac019
# ╠═a5793352-2a91-4cb4-96db-ef21973767f3
# ╟─b0efb7d6-cd96-4108-97e8-da8e5d00b65f
# ╠═f6b654c7-0395-422c-b49e-d8a771625903
# ╠═f493e015-6831-4344-9dfc-fdc6f04ada72
# ╟─6ae1e759-1f0b-4131-8a0e-d7693875e4e6
# ╠═336a696f-2545-4f88-928c-1aaed4100988
# ╠═ff508904-d8f4-406f-a530-877ac212d8d6
# ╟─0379f1a2-eaae-4c6d-8324-29b3b0f306ca
# ╠═8b6cf25c-31be-4f0f-b226-7d7137d24ed5
# ╟─eb21a633-a15b-4670-89e1-b4beb1d2f9b5
# ╠═0edfd386-6b70-4884-aba8-d1964d9cdc97
# ╟─d91aec6c-ab70-4538-ae16-3127ecc2a793
# ╠═8c51ef55-7e49-424f-85cf-315fa5e92836
# ╠═92694d37-2612-4a36-9809-6fe28811af07
# ╟─da150260-1d36-4a1d-9249-3abab7014ac6
# ╠═c75116f1-5e8a-438a-a349-047283ac701f
# ╠═fa2b4e05-e07b-408d-886d-3f375da26f35
# ╟─55452732-7f91-4493-85ef-6dba712f33b0
# ╠═026934ec-6f67-45f4-bfec-fc44ab6fa913
# ╟─648af4da-437f-4207-bb63-200b29ff1243
# ╠═a3b74a06-c25e-4ecc-bfb1-c669e735809b
# ╟─3c8b44de-2986-4cbb-aca3-fd88b78dee6b
# ╠═11a63b16-1f0d-49d4-8a15-0cb726e2627a
# ╟─b14e5668-fbc6-4a8b-b613-c37335abfa77
# ╠═b206e6a2-6b00-4863-aecb-833f17e9fc77
# ╠═5d3807b6-fdde-4758-8955-77816f2758c5
# ╟─9b00bed5-bf85-45ae-b768-3f4316852f38
# ╠═1c99a8a8-2f87-43b9-9316-1724e387ad3d
# ╠═2ff59f98-598b-444f-9be6-67b212ee4fe2
# ╟─fa3ee0f0-0f8c-4146-ae2c-622050a57999
# ╠═6579d3bb-bf7e-46c2-ae8b-11627be11b11
# ╟─e59cfa75-c127-4fde-bea2-af97f2857dcd
# ╠═952eeef1-f3bc-48bd-a974-21d4dea587aa
# ╟─6f7138ab-133a-4e2a-9a56-87754b368450
# ╠═16e7014f-d553-4a08-9dec-0e5f33946c3f
