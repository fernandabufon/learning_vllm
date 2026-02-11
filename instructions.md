# Como rodar vLLM (container + túnel SSH)

## 0) Pré-requisitos

* Docker + NVIDIA runtime funcionando na máquina remota (`nvidia-smi` ok).
* Você acessa a máquina remota via:

```bash
ssh aluno_iago@200.137.197.252 -p 25210
```

---

## 1) Subir o container com porta e volume

Crie/edite seu `container.sh`:

```bash
#!/bin/bash

docker run \
  -it \
  --name fer_corejur \
  --gpus all \
  --shm-size=60g \
  --memory=60g \
  --cpuset-cpus=0-1 \
  -p 127.0.0.1:8000:8000 \
  -v /home/aluno_iago/fefe/corejur:/workspace \
  -w /workspace \
  nvidia/cuda:12.1.1-runtime-ubuntu22.04 bash
```

> `-p 127.0.0.1:8000:8000` expõe a API **só no host local** (seguro) — ideal para túnel SSH.

Se o container já existir:

```bash
docker rm -f fer_corejur
bash container.sh
```

---

## 2) Instalar dependências dentro do container

Dentro do container:

```bash
apt-get update
apt-get install -y build-essential curl
```

> `build-essential` é necessário porque o PyTorch/Triton pode precisar compilar kernels (senão dá erro “Failed to find C compiler”).

---

## 3) Criar venv e instalar vLLM (dentro do container)

Dentro do container (recomendado usar venv local no volume):

```bash
cd /workspace
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install vllm
```

---

## 4) Rodar o servidor vLLM

Dentro do container:

```bash
source /workspace/.venv/bin/activate

vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096
```

Se der erro/instabilidade, rode mais “conservador”:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization 0.85
```

---

## 5) Conferir se está no ar (na máquina remota)

### Dentro do container:

```bash
curl -4 http://127.0.0.1:8000/v1/models
```

### No host (fora do container):

```bash
curl -4 http://127.0.0.1:8000/v1/models
```

---

## 6) Acessar do seu PC via túnel SSH

No seu PC (Windows/WSL), abra o túnel:

```bash
ssh -L 8000:127.0.0.1:8000 aluno_iago@200.137.197.252 -p 25210
```

Deixe esse terminal aberto.

> Se você vai rodar o client no **WSL**, abra o túnel no **WSL** também.
> Se vai rodar no **PowerShell**, abra o túnel no **PowerShell**.

---

## 7) Testar do seu PC

### Ver modelos:

```bash
curl -4 http://127.0.0.1:8000/v1/models
```

### Completions:

```bash
curl -4 http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "San Francisco is a",
    "max_tokens": 20,
    "temperature": 0
  }'
```

### Chat (melhor para Instruct):

```bash
curl -4 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role":"user","content":"Diga 1 frase sobre San Francisco."}],
    "max_tokens": 50,
    "temperature": 0
  }'
```

---

## 8) Problemas comuns

* **`Connection refused`**: porta não publicada (`-p ...`) ou vLLM não está rodando.
* **Timeout em `/v1/models`**: túnel aberto no lugar errado (Windows vs WSL) ou porta local ocupada.
* **`Failed to find C compiler`**: faltou `apt-get install build-essential`.
* **Baixou modelo de novo**: cache HF diferente no container (persistir cache resolve).

---

## 9) No online como API da OpenAI:
```from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",  
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
    temperature=0,
    max_tokens=80,
)

print(resp.choices[0].message.content)
```
