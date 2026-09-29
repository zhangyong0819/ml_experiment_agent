---
doc_id: transformers-loading
title: Local model files and imports
source: https://huggingface.co/docs/transformers/main_classes/model
topic: model_loading
error_type: PATH_ERROR
---

# Local model files and imports

Model loading paths

from_pretrained can load supported local model directories or model identifiers. Missing files, an unavailable cache and incompatible library imports have different causes. Inspect the exact exception and configured location; do not assume every loading failure is a network error. The toy classification project does not load a Transformer model. Multilingual E5 is used only by documentation retrieval, so an embedding download failure must not be reported as toy training failure.

