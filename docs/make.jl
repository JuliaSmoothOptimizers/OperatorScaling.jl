using Scaling
using Documenter

DocMeta.setdocmeta!(Scaling, :DocTestSetup, :(using Scaling); recursive = true)

makedocs(;
  modules = [Scaling],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Geoffroy Leconte <gleconte50@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/Scaling.jl/blob/{commit}{path}#{line}",
  sitename = "JSOTemplate.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/Scaling.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/Scaling.jl",
  push_preview = true,
  devbranch = "main",
)
