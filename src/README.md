```bash
scons
```

Buffer generation lifecycle. The inference engine will first:
1. Parse the onnx to the ml::Graph structure
2. The ml::TensorResourceManager will be used to create all the buffers from the initializers. This meaning the activations/weights and biases of each GraphNode.
3. Iterate through the sequence of GraphNodes. For each one, request to the ml::TensorResourceManager it's required tensor buffers. Note that in this case, the output buffer might not be created. If that it's the case, the ResourceManager creates that based on the given size of the tensor. The usage should be something like ml::TensorResourceManager::get(name, sizes).
4. The inference engine gets the storage buffers, binds them, pushes the constants of the operator and submits the task.
5. Once the previous task completed, goes to the next graph node. This is repeated until all graph nodes are traversed.
