# Dados sintéticos para ampliar datasets pequenos (PathMNIST)

Experimento para o artigo de IA 2: com a mesma CNN, comparar treinar só com uma fração pequena dos dados reais contra
essa fração + 2x de imagens sintéticas (CVAE ou cGAN) ou aumentadas de forma clássica, mais uma referência com 100% real.

## Estrutura

| Arquivo | O que faz |
|---|---|
| `config.py` | **Todos os hiperparâmetros** (sementes, fração real, proporção sintética, épocas, caminhos) |
| `dataset.py` | Carrega o PathMNIST, divide a validação (early stopping × avaliação no domínio do treino) e sorteia o subconjunto real por semente |
| `classifier.py` | CNN pequena + treino com early stopping + avaliação em dois conjuntos (acurácia e F1 macro) |
| `cvae.py` | VAE condicional convolucional (treino e geração) |
| `cgan.py` | DCGAN condicional (treino e geração) |
| `augment.py` | Aumento clássico (rotações de 90° e flip), cenário 4 |
| `quality.py` | Qualidade das sintéticas: consistência condicional, cor, nitidez e diversidade |
| `plots.py` | Figura comparativa real × CVAE × cGAN |
| `pilot_fraction.py` | **Piloto** para escolher a fração de dados reais (só baseline e referência, sem geradores) |
| `run_experiment.py` | **Script principal**: roda as sementes e salva um JSON por semente |
| `aggregate_results.py` | Gera as tabelas (cenários e qualidade), LaTeX e comparação pareada com o baseline |
| `utils.py` | Semente e one-hot |

## Avaliação em dois conjuntos

O treino e a validação do PathMNIST vêm do mesmo domínio, mas o teste oficial parece vir de outra fonte (mudança de domínio; confirme na documentação do MedMNIST). Por isso cada modelo é avaliado em:

- **Domínio do treino (avaliação principal):** o resto da validação oficial, que nunca entra no treino nem no early stopping.
- **Teste oficial (secundária):** mede a robustez à mudança de domínio.

O early stopping usa uma fração da validação (a mesma `FRAC_REAL`, estratificada e fixa); o restante é a avaliação no domínio do treino.

## Instalação (local, com GPU NVIDIA)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# instale o PyTorch conforme sua CUDA: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Ordem de uso

```bash
# 1) teste rápido de ponta a ponta (poucas épocas, 1 semente, pasta separada)
python run_experiment.py --seeds 0 --epochs-vae 3 --epochs-gan 3 --results-dir resultados_teste

# 2) piloto: escolhe a fração de dados reais (não treina geradores)
python pilot_fraction.py

# 3) defina FRAC_REAL em config.py (ou use --frac) e rode uma semente completa para medir o tempo
python run_experiment.py --seeds 0 --frac 0.02

# 4) experimento completo (5 sementes, definidas em config.py)
python run_experiment.py --frac 0.02

# 5) tabelas finais
python aggregate_results.py
```

Os resultados vão para `resultados/` (ou `--results-dir`, ou a variável `RESULTS_DIR`): `seed_N.json` (um por semente),
`grade_real_cvae_cgan.png`, `resumo.csv`, `resultados_por_semente.csv`, `qualidade_resumo.csv`,
`qualidade_por_semente.csv`, `cc_por_classe.csv` e, do piloto, `piloto_resumo.csv` e `piloto_por_semente.csv`.
Se o processo for interrompido, rode de novo: sementes já concluídas são puladas.

**Atenção:** não reaproveite a pasta do teste rápido nem a de outra fração. Os `seed_N.json` existentes são pulados, e um resultado de 3 épocas (ou de outra fração) seria confundido com o definitivo.

## Escolha da fração (piloto)

`pilot_fraction.py` treina o baseline em várias frações (padrão: 1%, 2%, 5% e 10%) e a referência (100% real), em 3 sementes, e mostra a **lacuna** (referência − baseline). Escolha a fração em que a lacuna é bem maior que o desvio entre sementes, **sem olhar o desempenho dos sintéticos** (escolher por quem ganha enviesaria o resultado). Registre essa regra e o piloto na metodologia. O piloto usa uma divisão de validação fixa (10%) para todas as frações, o que torna as comparações entre frações justas.

## Usando o Colab como GPU

Suba a pasta para o Google Drive e rode em um notebook com GPU T4:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd "/content/drive/MyDrive/artigo_ia2"
!pip -q install medmnist
import os
os.environ['DATA_DIR'] = '/content/drive/MyDrive/artigo_ia2_dados'
os.environ['RESULTS_DIR'] = '/content/drive/MyDrive/artigo_ia2_resultados'
!python pilot_fraction.py
```

Depois, `!python run_experiment.py --frac 0.02` e `!python aggregate_results.py`.

## O que ajustar e onde

- Menos tempo: reduza `SEEDS`, `EPOCHS_GAN` ou `EPOCHS_VAE` em `config.py`.
- Proporção ou fração: `SYN_RATIO` e `FRAC_REAL` (os nomes dos cenários se atualizam sozinhos).
- Arquitetura da CNN: classe `SmallCNN` em `classifier.py` (a mesma em todos os cenários).
- Arquitetura dos geradores: `cvae.py` e `cgan.py`.

## Qualidade das imagens sintéticas

Calculada ao final de cada semente, **sem treino adicional**. Em `quality.py`:

| Métrica | O que mede | Como ler |
|---|---|---|
| **CC** (consistência condicional) | % de sintéticas que o juiz classifica na classe pedida ao gerador | Compare com a acurácia do mesmo juiz em imagens reais |
| **Cor (W1)** | Distância de Wasserstein-1 entre histogramas de cada canal, em [0,1] | Menor é melhor; compare com o "piso" real × real |
| **Nitidez** | Variância do Laplaciano (tons de cinza) | Razão sintética/real: abaixo de 1 indica borrão |
| **Diversidade** | Distância L2 média entre pares da mesma classe (pixels) | Razão sintética/real: abaixo de 1 indica menos variedade (colapso de modo) |

- **Juiz:** a CNN treinada com 100% real da mesma semente. Com `--no-reference`, usa a do baseline (mais fraca, então a CC absoluta cai).
- **Referência real:** o resto da validação, nunca usado para treinar.
- **Piso de cor:** W1 entre o subconjunto real de treino e a referência real (o quanto duas amostras reais já diferem).
- **Limitações:** nenhuma métrica julga o realismo no sentido médico. CC alta não prova realismo. Nitidez e diversidade são medidas em pixels.

## Decisões metodológicas (para a seção de metodologia)

- Os geradores são treinados **apenas nos dados reais da fração**, nunca no treino completo (evita vazamento).
- O subconjunto real é estratificado por classe e muda a cada semente, então o desvio-padrão reflete a variação dos dados.
- Os sintéticos mantêm a distribuição de classes do subconjunto real (2 imagens por imagem real).
- Aumento clássico: cada imagem recebe 2 transformações distintas entre rotações de 90° e flip horizontal.
- O desvio-padrão é amostral (n-1). Com poucas sementes, o valor de p é apenas indicativo.
