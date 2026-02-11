## `vllm.SamplingParams` 

SamplingParams é o objeto que define como o modelo vai gerar (quantos tokens, temperatura, top_p, stops, etc.).

* Offline (Python vLLM): você passa SamplingParams(...) direto no llm.generate(...).
* API (servidor OpenAI-like): você não passa SamplingParams como classe; você manda os mesmos controles no JSON da requisição (ex.: temperature, top_p, max_tokens, stop, n).

* `n` — **1** — nº de sequências retornadas.
* `best_of` — **None** (vira `n`) — gera `best_of` e retorna as top `n`.
* `presence_penalty` — **0.0** — penaliza tokens já usados (incentiva novidade).
* `frequency_penalty` — **0.0** — penaliza repetição por frequência.
* `repetition_penalty` — **1.0** — >1 reduz repetição (prompt+gerado).
* `temperature` — **1.0** — aleatoriedade (0 = greedy).
* `top_p` — **1.0** — nucleus sampling.
* `top_k` — **-1** — limita top-k (-1 = sem limite).
* `min_p` — **0.0** — corta tokens muito improváveis (relativo ao melhor).
* `seed` — **None** — seed da geração.
* `stop` — **None** — strings que param a geração.
* `stop_token_ids` — **None** — token IDs que param a geração.
* `bad_words` — **None** — bloqueia sequências específicas.
* `ignore_eos` — **False** — ignora EOS e continua gerando.
* `max_tokens` — **16** — **máx tokens gerados** (causa “texto cortado”).
* `min_tokens` — **0** — mínimo antes de permitir stop/EOS.
* `logprobs` — **None** — retorna logprobs dos top tokens.
* `prompt_logprobs` — **None** — logprobs dos tokens do prompt.
* `detokenize` — **True** — retorna texto (não só ids).
* `skip_special_tokens` — **True** — remove tokens especiais do texto.
* `spaces_between_special_tokens` — **True** — adiciona espaços entre especiais.
* `logits_processors` — **None** — funções que alteram logits.
* `include_stop_str_in_output` — **False** — inclui `stop` no output.
* `truncate_prompt_tokens` — **None** — usa só os últimos *k* tokens do prompt.
* `output_kind` — **RequestOutputKind.CUMULATIVE** — formato do output (acumulado).
* `output_text_buffer_length` — **0** — buffer de texto (streaming).
* `guided_decoding` — **None** — guided decoding (gera conforme regra/schema).
* `logit_bias` — **None** — bias por token id (↑/↓ prob).
* `allowed_token_ids` — **None** — restringe geração a uma whitelist de ids.
