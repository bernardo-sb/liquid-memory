# End-to-end Text Ingestion Example

This example shows how to ingest text data into a vector database and query it.

The example use text-embedding-inference to get embeddings for the text data.

## Run Text Embedding Inference

To run Text Embedding Inference, you can use the following command: `text-embeddings-router --model-id BAAI/bge-large-en-v1.5  --port 8888` (model can be changed).

## Run vector database

To run Qdrant, you can use the following command: `docker run -p 6333:6333 qdrant/qdrant`

## Example workflow

1. Start qdrant client
2. Create collection
3. Embed text
4. Insert embeddings
5. Query for similar text
