using Documenter, ELMNIST

makedocs(;
    modules=[ELMNIST],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/rjww/ELMNIST.jl/blob/{commit}{path}#L{line}",
    sitename="ELMNIST.jl",
    authors="Robert Woods",
    assets=String[],
)

deploydocs(;
    repo="github.com/rjww/ELMNIST.jl",
)
