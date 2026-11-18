# Validação inicial da CI

A infraestrutura foi implementada sem modificar `src/`, os testes existentes,
`pyproject.toml`, `uv.lock` ou as configurações das ferramentas. As pendências
abaixo já existem no projeto e impedem considerar a primeira execução verde.

## Bloqueios preexistentes

- **Resolvido — Python 3.11:** o suporte foi descontinuado oficialmente; a
  versão mínima agora é Python 3.12. Isso resolve o bloqueio de coleta causado
  por imports de `typing.override`. A matriz Core passou de cinco para quatro
  ambientes, totalizando dez execuções de jobs por PR.
- **Static Quality / formatação:** `ruff format --check .` com Ruff 0.17.5
  (versão do lockfile) reprova quatro arquivos: `README.md`,
  `src/tyrannis/algorithm/egna.py`, `src/tyrannis/algorithm/gravitational.py` e
  `src/tyrannis/algorithm/psogsa.py`. Nenhuma correção automática foi aplicada.
- **Static Quality / tipagem:** basedpyright 1.40.2 (versão do lockfile), usando
  a `.venv` Python 3.12 local, reportou 59 erros e 2253 warnings na análise
  global. Um exemplo é `reportMissingTypeArgument` para `ndarray` em
  `src/tyrannis/algorithm/ant_colony.py:199`. Os dois novos scripts possuem zero
  diagnósticos. Os totais globais podem variar no runner Linux.

Reprodução das verificações estáticas no ambiente instalado pela CI:

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync basedpyright --pythonpath .venv/bin/python
```

No Windows, o argumento `--pythonpath` deve apontar para
`.venv/Scripts/python.exe`.

## Verificações locais concluídas

- actionlint 1.7.12: sintaxe, expressões e estrutura do workflow aprovadas.
- SHAs das três actions conferidos contra suas tags oficiais.
- `uv lock --check --offline` com uv 0.13.5: aprovado, sem alterar o lockfile.
- Ruff lint global: aprovado com a versão 0.17.5 do lockfile.
- Scripts da CI: Ruff, Ruff format e basedpyright aprovados tanto com as
  ferramentas globais quanto com as versões já instaladas na `.venv`.
- Cache: três testes opcionais aprovados; relatório JUnit real aprovado.
- Validador JUnit: 14 cenários verificados, incluindo sucesso, múltiplas suites,
  relatório ausente ou malformado, ausência de testes, skip de coleta, falhas,
  erros e contadores inválidos ou inconsistentes.
- Todos os sete arquivos que atualmente usam `pytest.mark.optional` estão
  selecionados pelos jobs Cache, Spark ou MPI.
- Wheel e sdist gerados pelo backend existente e aprovados pelo
  `twine==6.2.0 check --strict`.
- Cada artefato instalado sem extras nem dependências de desenvolvimento em
  uma `.venv` temporária independente. Ambos passaram no smoke da API pública,
  fora do checkout, com Python em modo isolado e origem do import verificada.

O build emitiu um aviso porque o uv 0.13.5 é diferente do intervalo do backend
`uv_build>=0.12.5,<0.13.0`. O backend declarado foi respeitado e ambos os builds
terminaram com sucesso; não houve alteração do `pyproject.toml`.

## Dependências da primeira execução no GitHub Actions

A suíte completa não foi executada localmente. Ainda precisam ser observados
nos runners os quatro ambientes Core, Spark com Java 17 e workers locais,
OpenMPI com três ranks e `MPI_THREAD_MULTIPLE`, restauração de cache,
cancelamento por concorrência e a consolidação real dos jobs pelo CI Gate.
O gate só aceita `success` em todos os seis grupos; as pendências acima não
foram suprimidas ou contornadas.

## Transição para Python 3.12 mínimo

`requires-python` passou a `>=3.12,<3.15`; o classifier, a matriz e o roadmap
de release deixaram de anunciar Python 3.11. `Mixed._evaluate` agora usa
`typing.override`, sem a suppression de compatibilidade anterior. A busca
direcionada não encontrou outros compromissos de suporte a 3.11; a referência
histórica acima foi preservada.

`uv lock` e `uv lock --check` passaram. Nenhuma versão dos pacotes restantes
mudou: foram removidos `tomli==2.4.1`, os marcadores de resolução para 3.11 e
wheels exclusivos dessa versão. NumPy 2.1.3 e SciPy 1.15.3 permaneceram.

Validação efetiva em Windows:

- Python 3.12.13: `uv sync --locked --dev` aprovado; coleta de 901 testes não
  opcionais, sete desmarcados; 11 testes de `tests/unit/space/test_mixed.py`
  aprovados. Os skips legítimos de coleta dos módulos Spark foram preservados.
- Python 3.13.2: ambiente temporário instalado com `uv sync --locked --dev
  --python 3.13`; coleta com os mesmos 901 testes e sete desmarcados aprovada.
- Python 3.14.6: instalação temporária com builds de NumPy/SciPy desabilitados
  falhou por ausência de distribuição binária de NumPy 2.1.3. O lockfile também
  não contém wheels CPython 3.14 de SciPy 1.15.3, tanto para Windows x64 quanto
  para Linux x64. Não houve coleta nessa versão; ela não está validada.
- Ruff lint e formatação de `mixed.py`: aprovados. basedpyright: um erro e
  43 warnings preexistentes; o erro é `reportMissingTypeArgument` em
  `_boundaries: list`, linha 13, já presente no relatório anterior à alteração.
  Nenhum diagnóstico foi introduzido pelo decorator.
- actionlint aprovado; quatro entradas Core e seis outros jobs, total dez.

O plano Linux x64 de `uv sync --locked --all-extras --dev --python 3.12
--python-platform x86_64-unknown-linux-gnu --dry-run` foi aprovado. Isso verifica
a seleção de pacotes, mas não substitui instalação, build ou execução no Linux.
Os wheels registrados cobrem NumPy, SciPy, pandas, pyarrow e mpi4py no Python
3.12 dos jobs opcionais. PySpark é distribuído como sdist e requer build;
as integrações reais Spark/MPI continuam dependendo dos runners. A ausência
de wheels 3.14 de pandas/pyarrow não afeta os jobs atuais, restritos a 3.12.

**Pendência independente — Core Tests / Python 3.14:** elevar a versão mínima
do Python não atualizou NumPy/SciPy. Proposta para tarefa separada: atualizar
seletivamente esses dois pacotes no lockfile para versões com suporte a 3.14
e wheels Linux/Windows, preservando os intervalos declarados, e repetir
instalação/coleta/testes. Não foi executado upgrade nesta tarefa. Sem isso, a
CI pode tentar compilar versões sem suporte adequado a 3.14.

As pendências de formatação e tipagem de Static Quality permanecem; nenhuma
correção fora do escopo foi aplicada. As verificações locais acima não
representam execução integral da suíte nem validação dos runners Linux.
