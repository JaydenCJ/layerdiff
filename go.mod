// layerdiff — compares two model checkpoints tensor by tensor: streamed
// hashes, numeric stats, and a changed-layer report, framework-free.
//
// version:    0.1.0
// author:     JaydenCJ
// license:    MIT
// repository: https://github.com/JaydenCJ/layerdiff
// keywords:   safetensors, checkpoint, diff, tensors, finetuning, model-merging, ml-tooling
//
// Zero runtime dependencies: standard library only.
module github.com/JaydenCJ/layerdiff

go 1.22
