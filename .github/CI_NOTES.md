# Validação inicial da CI

A infraestrutura foi implementada sem modificar `src/`, os testes existentes,
`pyproject.toml`, `uv.lock` ou as configurações das ferramentas. As pendências
abaixo já existem no projeto e impedem considerar a primeira execução verde.

## Bloqueios preexistentes

- **Core Tests / Python 3.11:** testes e helpers importam `override` diretamente
  de `typing`, que só o disponibiliza a partir do Python 3.12. Exemplos:
  `tests/_support/integration_probe.py:12`, `tests/_support/processors.py:8` e
  `tests/unit/core/test_space.py:6`. O comando `python3.11 -c "from typing import
  override"` foi reproduzido com um interpretador 3.11 local e gera `ImportError`.
  Isso impede a coleta nessa combinação. A matriz solicitada foi preservada.
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
nos runners os cinco ambientes Core, Spark com Java 17 e workers locais,
OpenMPI com três ranks e `MPI_THREAD_MULTIPLE`, restauração de cache,
cancelamento por concorrência e a consolidação real dos jobs pelo CI Gate.
O gate só aceita `success` em todos os seis grupos; as pendências acima não
foram suprimidas ou contornadas.
