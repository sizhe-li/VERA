import tensorflow as tf


def recursive_cast_to_numpy(obj):
    if isinstance(obj, tf.Tensor):
        if obj.dtype == tf.string:
            # Decode the string tensor to Python strings
            return obj.numpy().tolist() if obj.ndim > 0 else obj.numpy().decode("utf-8")
        else:
            # Convert non-string tensors to numpy arrays
            return obj.numpy()
    elif isinstance(obj, dict):
        # Recursively handle dictionary values
        return {key: recursive_cast_to_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Recursively handle list elements
        return [recursive_cast_to_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively handle tuple elements
        return tuple(recursive_cast_to_numpy(item) for item in obj)
    else:
        # Return the object as-is if it's not a tf.Tensor
        return obj
