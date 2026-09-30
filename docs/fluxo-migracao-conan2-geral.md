# Sequência de Migração para Conan 2 - Geral (v.1)

> Esse diagrama tem como propósito exibir o fluxo de migração dos repositórios dos projetos BrKin, AchillesBR, StratBR e SimBR para o Conan 2.

```mermaid
flowchart TD
    A["Receitas Conan 2<br/><small>s11524-modgeo-genesisplataforma-receitasconan2</small><br/><tiny>Qt/Qt5, Boost, DevKit, OpenInventor, terralib, GDAL, HDF5, SQLite, Gds, CGAL, Qwt, Eigen3, SWSolver, Gmsh, xerces-c, xsd, gtest, ninja, cmake, etc.</tiny>"]

    B["Genesis Framework<br/><small>s11524-modgeo-genesisplataforma-genesisframework</small>"]
    C["Geoconnector<br/><small>s11524-modgeo-genesisplugins-geoconnector</small>"]
    D["Genesis Extensions<br/><small>s11524-modgeo-genesisplugins-genesisextensions</small>"]

    E["Geologia<br/><small>s11524-modgeo-genesisplugins-geologia</small>"]
    F["BrKin<br/><small>s11899-modgeo-genesisplugins-geoquimica</small>"]

    G["Geoestatística<br/><small>s11436-modgeo-genesisplugins-geoestatistica</small>"]
    H["BSMIOX<br/><small>sibr-bsmiox</small>"]

    I["StratModeling<br/><small>s11436-modgeo-genesisplugins-modelagemestratigrafica</small>"]
    J["BSMIO<br/><small>sibr-bsmio</small>"]

    K["AchillesBR<br/><small>s11899-modgeo-genesisplugins-organicfacies</small>"]
    L["MeshGenerator<br/><small>sibr-meshgenerator</small>"]

    M["Fractal Analysis<br/><small>s11436-modgeo-genesisplugins-fractalanalysis</small>"]
    N["SimBR<br/><small>sibr-simbr</small>"]

    O["StratBR<br/><small>s11436-modgeo-genesisplugins-stratbr</small>"]

    Z["Standalone<br/><small>s11436-modgeo-genesis-standalone</small>"]

    A --> B --> C --> D
    D --> E & F

    E --> G & H
    H --> J --> L --> N --> Z

    G --> I
    I --> K
    K --> M & Z
    M --> O
    O --> Z

    F --> Z
```

## Legenda

- **Repositórios base para Conan 2** — camada de fundo (Receitas + Framework)
- **Já migrados** — plugins de plataforma já transportados
- **Próximo repositório para migração** — próximo da fila
- **Repositórios não migrados** — pendentes
- **Repositório para instalador compartilhado** — `Standalone`

---

*Última atualização: 30/09/2026*

> A ordem de migração apresentada neste diagrama foi determinada através de análise automatizada dos repositórios e ordenação por prioridade ou dependências críticas. Não representa necessariamente a ordem de dependências explícitas dos conanfiles de cada projeto.