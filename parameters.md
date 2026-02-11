## ModelConfig (modelo/tokenizer/precisão)

* `--tokenizer` (`tokenizer`) — **Default: None** — tokenizer HF (se None, usa o do modelo).
* `--tokenizer-mode` (`tokenizer_mode`) — **Default: auto** — modo do tokenizer (hf/slow/mistral/etc.).
* `--trust-remote-code / --no-trust-remote-code` (`trust_remote_code`) — **Default: False** — permite código remoto HF.
* `--dtype` (`dtype`) — **Default: auto** — dtype de pesos/ativações (fp16/bf16/fp32…).
* `--seed` (`seed`) — **Default: 0** — seed global (importante em TP).
* `--hf-config-path` (`hf_config_path`) — **Default: None** — caminho/nome de config HF alternativo.
* `--revision` (`revision`) — **Default: None** — revisão do modelo no HF Hub.
* `--code-revision` (`code_revision`) — **Default: None** — revisão do código do modelo no HF Hub.
* `--tokenizer-revision` (`tokenizer_revision`) — **Default: None** — revisão do tokenizer no HF Hub.
* `--max-model-len` (`max_model_len`) — **Default: None** — contexto (prompt+output); aceita `auto/-1` e sufixos k/m/g.
* `--quantization / -q` (`quantization`) — **Default: None** — método de quantização (ou lê do `quantization_config`).
* `--allow-deprecated-quantization / --no-allow-deprecated-quantization` (`allow_deprecated_quantization`) — **Default: False**.
* `--enforce-eager / --no-enforce-eager` (`enforce_eager`) — **Default: False** — força eager; desliga CUDA graph.
* `--enable-return-routed-experts / --no-enable-return-routed-experts` (`enable_return_routed_experts`) — **Default: False**.
* `--max-logprobs` (`max_logprobs`) — **Default: 20** — limite de logprobs retornados (cuidado com OOM).
* `--logprobs-mode` (`logprobs_mode`) — **Default: raw_logprobs** — retorna logits/logprobs raw vs processed.
* `--disable-sliding-window / --no-disable-sliding-window` (`disable_sliding_window`) — **Default: False**.
* `--disable-cascade-attn / --no-disable-cascade-attn` (`disable_cascade_attn`) — **Default: False**.
* `--skip-tokenizer-init / --no-skip-tokenizer-init` (`skip_tokenizer_init`) — **Default: False** — pula init tokenizer/detokenizer (espera `prompt_token_ids`).
* `--enable-prompt-embeds / --no-enable-prompt-embeds` (`enable_prompt_embeds`) — **Default: False** — permite `prompt_embeds` (atenção a formato).
* `--served-model-name` (`served_model_name`) — **Default: None** — nome(s) exposto(s) na API (serve).
* `--config-format` (`config_format`) — **Default: auto** — formato de config (hf/mistral/auto).
* `--hf-token` (`hf_token`) — **Default: None** — token HF (ou usa o do `huggingface-cli login`).
* `--hf-overrides` (`hf_overrides`) — **Default: {}** — overrides passados pro config HF.
* `--pooler-config` (`pooler_config`) — **Default: None** — config de pooling (modelos de embedding/pooling).

---

## CacheConfig (KV cache / memória)

* `--kv-cache-dtype` (`kv_cache_dtype`) — **Default: auto** — dtype do KV cache (inclui fp8 variantes).
* `--block-size` (`block_size`) — **Default: None** — tamanho do bloco de cache (depende da plataforma).
* `--gpu-memory-utilization` (`gpu_memory_utilization`) — **Default: 0.9** — fração de VRAM usada pelo executor (por instância).
* `--num-gpu-blocks-override` (`num_gpu_blocks_override`) — **Default: None** — override de blocos (teste/preempção).
* `--enable-prefix-caching / --no-enable-prefix-caching` (`enable_prefix_caching`) — **Default: None** — liga prefix caching.
* `--prefix-caching-hash-algo` (`prefix_caching_hash_algo`) — **Default: sha256** — hash do prefix caching (xxhash etc.).
* `--cpu-offload-gb` (`cpu_offload_gb`) — **Default: 0** — offload CPU por GPU (aumenta “VRAM virtual”).
* `--calculate-kv-scales / --no-calculate-kv-scales` (`calculate_kv_scales`) — **Default: False** — calcula escalas se KV cache for fp8.
* `--kv-sharing-fast-prefill / --no-kv-sharing-fast-prefill` (`kv_sharing_fast_prefill`) — **Default: False**.

### Mamba cache (se aplicável)

* `--mamba-cache-dtype` (`mamba_cache_dtype`) — **Default: auto**.
* `--mamba-ssm-cache-dtype` (`mamba_ssm_cache_dtype`) — **Default: auto**.
* `--mamba-block-size` (`mamba_block_size`) — **Default: None**.
* `--mamba-cache-mode` (`mamba_cache_mode`) — **Default: (não aparece completo no trecho)**.

---

## SchedulerConfig (batching/throughput)

* `--max-num-batched-tokens` (`max_num_batched_tokens`) — **Default: None** — teto de tokens por iteração (ajuste essencial pra throughput).
* `--max-num-seqs` (`max_num_seqs`) — **Default: None** — teto de sequências por iteração.
* `--max-num-partial-prefills` (`max_num_partial_prefills`) — **Default: 1** — chunked prefill: quantas sequências parcialmente prefilladas ao mesmo tempo.
* `--max-long-partial-prefills` (`max_long_partial_prefills`) — **Default: 1** — chunked prefill: limite pra prompts “longos”.
* `--long-prefill-token-threshold` (`long_prefill_token_threshold`) — **Default: 0** — define o que é “longo”.
* `--scheduling-policy` (`scheduling_policy`) — **Default: fcfs** — fcfs vs priority.
* `--enable-chunked-prefill / --no-enable-chunked-prefill` (`enable_chunked_prefill`) — **Default: None** — permite chunking no prefill (em “pedaços”).
* `--disable-chunked-mm-input / --no-disable-chunked-mm-input` (`disable_chunked_mm_input`) — **Default: False** — evita fatiar item multimodal no chunked prefill (V1).
* `--scheduler-cls` (`scheduler_cls`) — **Default: vllm.v1.core.sched.scheduler.Scheduler** — classe do scheduler.

---

## ParallelConfig (multi-GPU / multinode)

* `--distributed-executor-backend` (`distributed_executor_backend`) — **Default: None** — backend mp/ray/uni/external_launcher.
* `--pipeline-parallel-size / -pp` (`pipeline_parallel_size`) — **Default: 1**.
* `--tensor-parallel-size / -tp` (`tensor_parallel_size`) — **Default: 1**.
* `--master-addr` (`master_addr`) — **Default: 127.0.0.1**.
* `--master-port` (`master_port`) — **Default: 29501**.
* `--nnodes / -n` (`nnodes`) — **Default: 1**.
* `--node-rank / -r` (`node_rank`) — **Default: 0**.

### Context parallel (DCP/PCP)

* `--decode-context-parallel-size / -dcp` (`decode_context_parallel_size`) — **Default: 1**.
* `--dcp-kv-cache-interleave-size` (`dcp_kv_cache_interleave_size`) — **Default: 1**.
* `--cp-kv-cache-interleave-size` (`cp_kv_cache_interleave_size`) — **Default: 1**.
* `--prefill-context-parallel-size / -pcp` (`prefill_context_parallel_size`) — **Default: 1**.

### Data parallel (DP)

* `--data-parallel-size / -dp` (`data_parallel_size`) — **Default: 1**.
* `--data-parallel-rank / -dpn` (`data_parallel_rank`) — **Default: None** — habilita modo external LB quando setado.
* `--data-parallel-start-rank / -dpr` (`data_parallel_start_rank`) — **Default: None**.
* `--data-parallel-size-local / -dpl` (`data_parallel_size_local`) — **Default: None**.
* `--data-parallel-address / -dpa` (`data_parallel_address`) — **Default: None**.
* `--data-parallel-rpc-port / -dpp` (`data_parallel_rpc_port`) — **Default: None**.
* `--data-parallel-backend / -dpb` (`data_parallel_backend`) — **Default: mp**.
* `--data-parallel-hybrid-lb / --no-data-parallel-hybrid-lb / -dph` (`data_parallel_hybrid_lb`) — **Default: False** (online).
* `--data-parallel-external-lb / --no-data-parallel-external-lb / -dpe` (`data_parallel_external_lb`) — **Default: False** (online).

### MoE / Expert parallel / overlap

* `--enable-expert-parallel / --no-enable-expert-parallel / -ep` (`enable_expert_parallel`) — **Default: False**.
* `--all2all-backend` (`all2all_backend`) — **Default: allgather_reducescatter** — backend de comunicação EP.
* `--enable-dbo / --no-enable-dbo` (`enable_dbo`) — **Default: False** — dual batch overlap.
* `--ubatch-size` (`ubatch_size`) — **Default: 0** — microbatch/ubatch.
* `--dbo-decode-token-threshold` (`dbo_decode_token_threshold`) — **Default: 32**.
* `--dbo-prefill-token-threshold` (`dbo_prefill_token_threshold`) — **Default: 512**.
* `--disable-nccl-for-dp-synchronization / --no-disable-nccl-for-dp-synchronization` (`disable_nccl_for_dp_synchronization`) — **Default: None** (auto).
* `--enable-eplb / --no-enable-eplb` (`enable_eplb`) — **Default: False** — load balancing EP.
* `--eplb-config` (`eplb_config`) — **Default: EPLBConfig(...)** — config EP LB.
* `--expert-placement-strategy` (`expert_placement_strategy`) — **Default: linear**.

---

## Worker / loading / all-reduce

* `--max-parallel-loading-workers` (`max_parallel_loading_workers`) — **Default: None** — limita loaders paralelos (evita OOM RAM).
* `--ray-workers-use-nsight / --no-ray-workers-use-nsight` (`ray_workers_use_nsight`) — **Default: False**.
* `--disable-custom-all-reduce / --no-disable-custom-all-reduce` (`disable_custom_all_reduce`) — **Default: False** — cai pra NCCL.
* `--worker-cls` (`worker_cls`) — **Default: auto** — classe de worker.
* `--worker-extension-cls` (`worker_extension_cls`) — **Default: ""** — extensão dinâmica do worker.

---

## Multimodal (MM)

* `--mm-processor-cache-gb` (`mm_processor_cache_gb`) — **Default: 4** — cache do pré-processador/mapper MM (duplica por processo).
* `--mm-processor-cache-type` (`mm_processor_cache_type`) — **Default: lru** — lru vs shm.
* `--mm-shm-cache-max-object-size-mb` (`mm_shm_cache_max_object_size_mb`) — **Default: 128**.
* `--mm-encoder-only / --no-mm-encoder-only` (`mm_encoder_only`) — **Default: False**.
* `--mm-encoder-tp-mode` (`mm_encoder_tp_mode`) — **Default: weights** — TP por pesos vs por dados.
* `--mm-encoder-attn-backend` (`mm_encoder_attn_backend`) — **Default: None** — override backend attention no encoder.
* `--interleave-mm-strings / --no-interleave-mm-strings` (`interleave_mm_strings`) — **Default: False**.
* `--skip-mm-profiling / --no-skip-mm-profiling` (`skip_mm_profiling`) — **Default: False** — pula profiling MM no init.
* `--video-pruning-rate` (`video_pruning_rate`) — **Default: None**.
* `--allowed-local-media-path` (`allowed_local_media_path`) — **Default: ""** — permite ler mídia local (risco).
* `--allowed-media-domains` (`allowed_media_domains`) — **Default: None** — restringe domínios de mídia por URL.

---

## LoRAConfig

* `--enable-lora / --no-enable-lora` (`enable_lora`) — **Default: None**.
* `--max-loras` (`max_loras`) — **Default: 1**.
* `--max-lora-rank` (`max_lora_rank`) — **Default: 16**.
* `--lora-dtype` (`lora_dtype`) — **Default: auto**.
* `--enable-tower-connector-lora / --no-enable-tower-connector-lora` (`enable_tower_connector_lora`) — **Default: False**.
* `--max-cpu-loras` (`max_cpu_loras`) — **Default: None**.
* `--fully-sharded-loras / --no-fully-sharded-loras` (`fully_sharded_loras`) — **Default: False**.
* `--default-mm-loras` (`default_mm_loras`) — **Default: None** — mapa modalidade→path (MM).

---

## ObservabilityConfig (métricas/tracing)

* `--show-hidden-metrics-for-version` (`show_hidden_metrics_for_version`) — **Default: None**.
* `--otlp-traces-endpoint` (`otlp_traces_endpoint`) — **Default: None**.
* `--collect-detailed-traces` (`collect_detailed_traces`) — **Default: None** — coleta detalhada (pode impactar performance).
* `--kv-cache-metrics / --no-kv-cache-metrics` (`kv_cache_metrics`) — **Default: False** — métricas de residência do KV cache.
* `--kv-cache-metrics-sample` (`kv_cache_metrics_sample`) — **Default: 0.01**.
* `--cudagraph-metrics / --no-cudagraph-metrics` (`cudagraph_metrics`) — **Default: False**.
* `--enable-layerwise-nvtx-tracing / --no-enable-layerwise-nvtx-tracing` (`enable_layerwise_nvtx_tracing`) — **Default: False**.
* `--enable-mfu-metrics / --no-enable-mfu-metrics` (`enable_mfu_metrics`) — **Default: False**.
* `--enable-logging-iteration-details / --no-enable-logging-iteration-details` (`enable_logging_iteration_details`) — **Default: False**.

---

## Compilation / Attention / Structured outputs / Profiling

* `--compilation-config` (`compilation_config`) — **Default: {…}** — config de compile/inductor/cudagraph (JSON).
* `--attention-config / -ac` (`attention_config`) — **Default: AttentionConfig(...)** — backend/flash-attn/etc.
* `--additional-config` (`additional_config`) — **Default: {}** — configs extras por plataforma (hashable).
* `--structured-outputs-config` (`structured_outputs_config`) — **Default: StructuredOutputsConfig(...)**.
* `--reasoning-parser` (`reasoning_parser`) — **Default: ""** — parser de reasoning para formato OpenAI.
* `--reasoning-parser-plugin` (`reasoning_parser_plugin`) — **Default: ""** — plugin de parser dinâmico.
* `--profiler-config` (`profiler_config`) — **Default: ProfilerConfig(...)**.
* `--optimization-level` (`optimization_level`) — **Default: 2** — -O0…-O3 (tradeoff startup vs perf).
* `--attention-backend` (`attention_backend`) — **Default: None** — override do backend de attention.

---

## AsyncEngineArgs (principalmente online)

* `--enable-log-requests / --no-enable-log-requests` (`enable_log_requests`) — **Default: False** — loga requests.
* `--disable-log-requests / --no-disable-log-requests` (`disable_log_requests`) — **[DEPRECATED]** — desliga logs de request.
