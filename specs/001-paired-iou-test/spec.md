# Feature Specification: Paired Per-Image IoU Test

**Feature Branch**: `001-paired-iou-test`

**Created**: 2026-11-06

**Status**: Draft

**Input**: User description: "Novo experimento para realizar um teste pareado de acordo com docs/spec-11/projeto de experimentos com imagens sismicas.tex - EXPERIMENTO A. Usando as redes pre treinadas. Uma rede somente com dados reais outra com reais + sinteticos. Os parametros da rede treinada com dados reais estao em resultsHPC/scenario_A_seed42_nreal1200_train/best_model.pth. Rede treinada com dados aumentados em resultsHPC/scenario_B_seed42_nreal1200_train_pairs1600_seismic_ns1200/best_model.pth. O comportamento do codigo para realizar o teste pareado deve ser ajustavel por parametros da linha de comando, permitindo executar o experimento anterior sem novas modificacoes no codigo. O mesmo conjunto de teste sera usado, mas as IoUs serao medidas por imagem."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Executar comparação pareada (Priority: P1)

Como pesquisador, quero avaliar os checkpoints já treinados com dados reais e com dados reais mais sintéticos nas mesmas imagens reais de teste, para medir a diferença de desempenho por imagem sem treinar novamente os modelos.

**Why this priority**: A inferência pareada por imagem é o objetivo central do Experimento A e permite testar se o modelo aumentado melhora sobre a base nas mesmas amostras.

**Independent Test**: Executar a análise com os dois checkpoints válidos e o conjunto canônico de teste; confirmar que cada imagem tem uma IoU da base, uma IoU do modelo aumentado e uma diferença pareada.

**Acceptance Scenarios**:

1. **Given** dois checkpoints compatíveis e um diretório de teste com imagens e máscaras, **When** o pesquisador executa a análise, **Then** os dois modelos avaliam as mesmas imagens e o processo não realiza treinamento.
2. **Given** uma amostra de teste identificada por nome-base de arquivo, **When** as métricas são calculadas, **Then** a IoU de cada modelo e sua diferença ficam associadas ao mesmo identificador, independentemente da ordem de listagem dos arquivos.
3. **Given** arquivos ausentes, IDs duplicados ou conjuntos de IDs diferentes entre imagens e máscaras, **When** a análise valida os dados, **Then** ela informa o problema e não produz uma comparação pareada parcial silenciosa.

---

### User Story 2 - Interpretar evidência estatística (Priority: P1)

Como pesquisador, quero receber as estatísticas pareadas e os diagnósticos definidos no Experimento A, para avaliar significância, magnitude e distribuição dos ganhos sem calcular métricas manualmente.

**Why this priority**: O contraste estatístico por imagem é necessário para responder à hipótese do experimento e comunicar tanto incerteza quanto relevância prática.

**Independent Test**: Comparar a saída com vetores conhecidos de IoU base e aumentada e confirmar direção do teste, estatísticas descritivas, efeito e conclusão conforme o alfa configurado.

**Acceptance Scenarios**:

1. **Given** IoUs válidas para os mesmos IDs, **When** a análise estatística termina, **Then** ela reporta o teste de Wilcoxon pareado com alternativa configurável, estatística, p-valor, alfa e decisão estatística correspondente.
2. **Given** diferenças pareadas positivas, negativas e nulas, **When** as estatísticas são resumidas, **Then** a saída inclui mediana e IQR de cada modelo, mediana da diferença, proporção de pares favoráveis e correlação bisserial por postos.
3. **Given** diferenças pareadas calculadas, **When** os diagnósticos são gerados, **Then** a saída apresenta Shapiro-Wilk, assimetria, inspeção gráfica da distribuição das diferenças e uma visualização da comparação pareada.
4. **Given** distribuição assimétrica ou resultado não significativo, **When** a conclusão é exibida, **Then** ela não afirma superioridade nem interpreta o Wilcoxon como teste de mediana sem ressalvar a hipótese de simetria.

---

### User Story 3 - Ajustar e reproduzir a execução pela linha de comando (Priority: P1)

Como pesquisador, quero configurar caminhos e parâmetros experimentais pela linha de comando, para repetir o Experimento A ou usar outros checkpoints e conjuntos de teste sem editar o código.

**Why this priority**: A execução reproduzível e ajustável é uma exigência explícita e evita alterações manuais que poderiam mudar o protocolo sem registro.

**Independent Test**: Executar a análise com os checkpoints informados e parâmetros padrão; depois trocar um caminho, limiar, alfa ou alternativa por argumento e confirmar que somente o comportamento correspondente muda.

**Acceptance Scenarios**:

1. **Given** os argumentos padrão e os arquivos do Experimento A, **When** o pesquisador inicia a execução, **Then** são usados os checkpoints indicados, o conjunto canônico e a configuração padrão do protocolo.
2. **Given** caminhos de checkpoints, diretório de teste ou saída diferentes, **When** o pesquisador os fornece na linha de comando, **Then** a execução usa esses caminhos sem alteração no código.
3. **Given** limiar, alfa e alternativa válidos fornecidos na linha de comando, **When** a análise é executada, **Then** as métricas binárias e a inferência estatística usam os valores fornecidos e registram esses valores nos resultados.
4. **Given** argumento ausente ou fora do intervalo permitido, **When** a execução valida os parâmetros, **Then** ela apresenta uma mensagem acionável e encerra antes de reportar resultados inválidos.

---

### Edge Cases

- Um checkpoint informado não existe, não pode ser carregado ou não corresponde à arquitetura esperada.
- O diretório de teste não contém a estrutura de imagens e máscaras esperada, ou contém pares incompletos.
- Os IDs das imagens e máscaras não coincidem, há IDs duplicados, ou os modelos não recebem exatamente o mesmo conjunto de IDs.
- Uma máscara de referência não contém pixels positivos; a métrica segue a convenção atual do projeto para união vazia, que produz IoU zero quando a união é zero.
- Todas as diferenças são zero ou há diferenças zero: os pares continuam na tabela por imagem e o relatório identifica quantos zeros houve; o efeito não deve ser apresentado como ganho.
- O alfa não está entre 0 e 1, o limiar não está no intervalo válido, ou a alternativa estatística não é suportada.
- O teste de Shapiro-Wilk não pode ser calculado para a amostra disponível; a limitação é registrada sem impedir o Wilcoxon se os pares forem válidos.
- A distribuição das diferenças é assimétrica; o relatório ressalva a interpretação do Wilcoxon como teste de mediana.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A ferramenta MUST avaliar os checkpoints da rede base (treinada somente com dados reais) e da rede aumentada (treinada com dados reais e sintéticos) sem iniciar treinamento.
- **FR-002**: A linha de comando MUST permitir informar os caminhos dos dois checkpoints, o diretório do conjunto de teste e o diretório de saída. Os padrões do Experimento A MUST apontar para `resultsHPC/scenario_A_seed42_nreal1200_train/best_model.pth` e `resultsHPC/scenario_B_seed42_nreal1200_train_pairs1600_seismic_ns1200/best_model.pth`; o padrão do teste MUST ser o conjunto canônico documentado pelo projeto.
- **FR-003**: A linha de comando MUST permitir configurar limiar de binarização, alfa do teste, alternativa estatística, tamanho de lote de inferência e dispositivo de execução; valores padrão MUST reproduzir o protocolo do projeto, incluindo limiar 0,5, alfa 0,05, alternativa `greater` e seleção automática do dispositivo.
- **FR-004**: A execução MUST usar um único conjunto de teste para os dois modelos, associar imagem e máscara pelo ID e produzir comparação apenas quando todos os IDs forem únicos e correspondentes nos dois modelos.
- **FR-005**: A inferência MUST aplicar aos dois modelos o mesmo pré-processamento compatível com o treinamento: entrada grayscale, normalização em [0,1], dimensão de entrada de 128×128 e limiar configurado para binarizar as predições. O redimensionamento de máscaras MUST preservar rótulos binários.
- **FR-006**: A ferramenta MUST calcular IoU individual para cada imagem e modelo antes de qualquer resumo, e MUST calcular a diferença pareada como IoU aumentada menos IoU base.
- **FR-007**: A análise MUST executar o Wilcoxon signed-rank pareado com alternativa configurável; o padrão MUST testar a hipótese direcional de IoU aumentada maior que IoU base. O resultado MUST incluir estatística, p-valor, alfa, direção da hipótese e decisão sem alegar ganho quando não sustentado pelos dados.
- **FR-008**: O resumo MUST reportar mediana e IQR de IoU para cada modelo, mediana das diferenças, percentual de pares em que o modelo aumentado foi melhor, soma dos postos positivos e negativos, correlação bisserial por postos e quantidade de diferenças zero.
- **FR-009**: A análise MUST apresentar o diagnóstico Shapiro-Wilk das diferenças, assimetria e gráficos da distribuição das diferenças e da comparação pareada. O diagnóstico de normalidade MUST NOT substituir automaticamente o teste Wilcoxon especificado.
- **FR-010**: O diretório de saída MUST conter uma tabela por imagem com ID, IoU base, IoU aumentada e diferença, além de um resumo estatístico que registre os caminhos de entrada e parâmetros da execução.
- **FR-011**: Falhas de leitura de checkpoints, dados inválidos, pareamento incompleto ou argumentos inválidos MUST resultar em erro explícito e impedir a publicação de um resumo que aparente conter comparação válida.

### Key Entities *(include if feature involves data)*

- **Modelo Base**: checkpoint treinado somente com dados reais; referência da comparação pareada.
- **Modelo Aumentado**: checkpoint treinado com dados reais e sintéticos; modelo sob avaliação comparativa.
- **Par de Teste**: imagem real, máscara de referência e ID único compartilhados pela avaliação dos dois modelos.
- **Registro por Imagem**: ID, IoU do modelo base, IoU do modelo aumentado e diferença entre essas IoUs.
- **Resumo Pareado**: métricas descritivas, estatística inferencial, tamanho de efeito, diagnósticos e configuração da execução.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Para o conjunto canônico de 800 imagens, a saída contém exatamente 800 registros pareados válidos, um por ID, sem imagens avaliadas por apenas um dos modelos.
- **SC-002**: Para cada registro, as duas IoUs e a diferença correspondem à mesma imagem e máscara; os resultados podem ser recalculados a partir da tabela por imagem.
- **SC-003**: Um pesquisador consegue executar o Experimento A com os dois checkpoints indicados e o conjunto de teste canônico somente por argumentos de linha de comando, sem modificar arquivos de código.
- **SC-004**: O resumo apresenta todos os resultados estatísticos e parâmetros exigidos pelo protocolo, permitindo verificar o p-valor contra o alfa selecionado e interpretar o tamanho/direção do efeito.
- **SC-005**: Ao substituir qualquer caminho, limiar, alfa ou alternativa válida pela linha de comando, o relatório registra a configuração efetiva e a execução usa essa configuração.
- **SC-006**: Um pesquisador consegue identificar em uma execução se há evidência estatística de melhora, a magnitude observada e as ressalvas de distribuição, sem inferir melhora quando o teste não a sustenta.

## Assumptions

- O conjunto canônico é o test set real de 800 imagens documentado em `/var/tmp/cym7/datasets/subset_split/test`; `--test-dir` pode substituí-lo para outros ambientes.
- Os dois checkpoints fornecidos são os padrões para reproduzir este Experimento A e podem ser substituídos por argumentos de linha de comando.
- O caminho de saída padrão é `resultsHPC/paired_iou_test_a`; ele pode ser substituído pela linha de comando.
- O pré-processamento e a definição de IoU seguem o treinamento existente: threshold 0,5 por padrão e denominador com epsilon, portanto uma união vazia resulta em IoU zero para manter comparabilidade.
- O teste estatístico padrão é unilateral (`greater`) com alfa 0,05, conforme o documento do Experimento A. A alternativa pode ser ajustada pela linha de comando, sem mudar o padrão documentado.
- Shapiro-Wilk e assimetria são diagnósticos; o Wilcoxon continua sendo o teste especificado. Uma interpretação como diferença de medianas depende de distribuição das diferenças aproximadamente simétrica.
- O experimento compara uma execução/checkpoint por cenário; não estima variação entre seeds.
- O objetivo é gerar análise e artefatos estatísticos reproduzíveis; atualizar manuscrito, resposta a revisores ou executar novo treinamento está fora do escopo.
