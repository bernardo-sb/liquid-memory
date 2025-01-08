# Image Text Embeddings Example

This example demonstrates how to work with both image and text embeddings using Liquid Memory. It showcases the following capabilities:

1. Image Embedding Generation
2. Text Description Generation from Images (using LLMs)
3. Text Embedding Generation
4. Multi-vector Storage and Retrieval

**Run the example:**
```bash
cargo run --examples image_text_embeddings
```

## Prerequisites

- A running instance of Text Embedding Inference server on port 8000 (for images) and 8888 (for text)
- A running Qdrant instance on port 6334
- Optional: Ollama running on port 11434 (for image-to-text conversion)

## Features

### Image Processing
- Loads images and converts them to base64 format
- Generates embeddings for images using Text Embedding Inference

### Text Processing
- Supports image-to-text conversion using LLMs (commented example using Ollama)
- Generates text embeddings using Text Embedding Inference

### Vector Storage
- Creates multi-vector collections in Qdrant
- Stores both image and text embeddings
- Supports similarity search across both modalities
