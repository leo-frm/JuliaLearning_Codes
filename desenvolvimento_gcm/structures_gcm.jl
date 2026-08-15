# Estruturas do Generalized Cell Mapping
# Reaproveita BasinRegion, BasinProblem, Attractor e BasinResult de structures.jl

mutable struct GcmProblem # Envolve um BasinProblem acrescentando o parâmetro de amostragem
    base :: BasinProblem #Problema base, idêntico ao usado no SCM
    samples_per_side :: Int64 #Número S de amostras por dimensão; gera S^d amostras por célula
end

# Mapa célula-a-célula generalizado, no formato dos três arranjos de Hsu (§11.1):
# I(z) número de imagens, C(z,i) i-ésima imagem, P(z,i) probabilidade de transição.
# Armazenado em CSR: as imagens da k-ésima célula da faixa ocupam
# images[offsets[k] : offsets[k+1]-1]. Guardam-se contagens inteiras em vez de
# probabilidades, já que o denominador (samples_per_cell) é constante.
mutable struct GeneralizedCellMap
    cell_range :: UnitRange{Int64} #Faixa de células coberta; 1:N no mapa completo, sub-faixa nos parciais
    offsets :: Vector{Int32} #Índices de início da lista de imagens de cada célula
    images :: Vector{Int32} #Células-destino concatenadas; -1 designa a célula sumidouro (divergência)
    counts :: Vector{Int32} #Número de amostras que chegou a cada célula-destino
    computed :: BitVector #Marcação de células já processadas
    samples_per_cell :: Int64 #S^d, denominador das probabilidades
    region :: BasinRegion
end

# Construtor: inicializa mapeamento vazio. As listas crescem conforme as células
# são processadas em ordem crescente dentro de cell_range.
function GeneralizedCellMap(
    region :: BasinRegion,
    n_samples :: Int64;
    cell_range :: UnitRange{Int64} = 1:prod(region.elements),
)
    offsets = Vector{Int32}(undef, length(cell_range) + 1)
    offsets[1] = 1

    return GeneralizedCellMap(
        cell_range,
        offsets,
        Int32[],
        Int32[],
        falses(length(cell_range)),
        n_samples,
        region,
    )
end