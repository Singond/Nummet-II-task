include("schrodinger.jl")

if !isdir("results")
	mkdir("results/")
end

for k in 1:24
	savefig(heatmap(x, y, erg[:,:,k], ratio=1),
		@sprintf "results/eig-gauss-%02d.pdf" k)
	savefig(heatmap(x, y, erb[:,:,k], ratio=1),
		@sprintf "results/eig-benzene-%02d.pdf" k)
end
