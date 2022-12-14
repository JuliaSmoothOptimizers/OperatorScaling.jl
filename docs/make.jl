using OperatorScaling
using Documenter

makedocs(;
  modules = [OperatorScaling],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Geoffroy Leconte <gleconte50@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/OperatorScaling.jl/blob/{commit}{path}#{line}",
  sitename = "OperatorScaling.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/OperatorScaling.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/OperatorScaling.jl",
  push_preview = true,
  devbranch = "main",
)
