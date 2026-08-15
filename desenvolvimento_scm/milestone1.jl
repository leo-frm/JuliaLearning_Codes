# Constrói o mapa célula‑a‑célula integrando um ponto de cada célula
function build_scmap(bp :: BasinProblem)
    scm = SimpleCellMap(bp.region)
    total_cells = prod(bp.region.elements)

    # Define o problema de EDO para integração de cada centro de célula
    ode_problem = ODEProblem(
        bp.f,
        zeros(Float64, length(bp.region.elements)),
        (0.0, bp.period * bp.maximum_cycles),
        bp.params,
    )

    # Cria integrador
    integrator = init(
        ode_problem;
        dense=false,
        save_everystep=false,
        save_start=false,
        maxiters=1e10,
    )
    
    # Integra cada célula uma vez e registra seu destino
    for cell in 1:total_cells
        scm.target[cell] = compute_single_cell_mapping(cell, integrator, bp)
        scm.computed_cells += 1 # computed_cells? paralelizavel no futuro -> passa um range
    end

    return scm
end