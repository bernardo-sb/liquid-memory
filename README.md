# Liquid Memory

Liquid Memory is an opinionated image and text ingestion and search framework.

**Vector store:**
- [Qdrant](https://qdrant.tech)

**Embedding clients:**
- OpenAI
- Text Embedding Inference (https://github.com/huggingface/text-embedding-inference)
- Image Embedding Inference ([in this repo](./image_embedding_inference/README.md))

**LLM clients:**
- OpenAI (https://platform.openai.com/docs/overview)
- Anthropic (https://docs.anthropic.com/en/api/getting-started)
- Ollama (via OpenAI spec)

## Run Qdrant

To run Qdrant, you can use the following command: `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`

Access the Qdrant UI at http://localhost:6333/dashboard

API Reference: https://api.qdrant.tech/api-reference

## Run Text Embedding Inference

To run Text Embedding Inference, you can use the following command: `text-embeddings-router --model-id BAAI/bge-large-en-v1.5  --port 8888`

## Running Examples

To run examples, you can use the following command: `cargo run --example <example_name>`. See the examples folder for available examples.

## Running Tests

To run tests, you can use the following command: `cargo test`
Qdrant tets are not mocked, so it requires a running Qdrant instance.
