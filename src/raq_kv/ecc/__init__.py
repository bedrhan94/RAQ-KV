"""Error-detecting and error-correcting codes over the stored KV representation.

Kept free of model and quantization concepts so the same implementation can produce golden vectors for an RTL
port later (plan §38-39): a codec sees bit matrices, nothing else.
"""
