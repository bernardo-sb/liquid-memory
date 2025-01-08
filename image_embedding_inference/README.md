# Image Embedding Inference (IEI)

This project provides a simple API to get embeddings for images using a pre-trained model. It has the same data contract as [text-embedding-inference](https://github.com/huggingface/text-embedding-inference).

## Supported Models

Image Embedding Inference supports multiple models from Huggingface. To select a model, set the environment variable `IMAGE_EMBEDDING_MODEL` to one of the following:

- `google/vit-base-patch16-224`
- `facebook/deit-base-patch16-224`
- `microsoft/beit-base-patch16-224`
- `facebook/convnext-tiny-224`
- `facebook/convnext-base-224`
- `microsoft/swin-tiny-patch4-window7-224`
- `microsoft/swin-base-patch4-window7-224`
- `nateraw/vit-base-beans`

If no model is specified, `google/vit-base-patch16-224` will be used.

## Setup

### Docker

Build the Docker image using the following command: `docker build -t image-embedding-inference .`

To run Image Embedding Inference in a Docker container, you can use the following command: `docker run -p 8000:8000 -e IMAGE_EMBEDDING_MODEL=google/vit-base-patch16-224 image-embedding-inference`

### Python

Install the dependencies using the following command: `uv pip install .`.

## Run Image Embedding Inference

To run Image Embedding Inference, you can use the following command: `IMAGE_EMBEDDING_MODEL=google/vit-base-patch16-224 gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --preload --bind 0.0.0.0:8000`

You can run an exapmple request using `python examples/request_example.py`