// Command make-demo fabricates a deterministic pair of demo checkpoints
// so layerdiff can be exercised without downloading real weights:
//
//	go run ./examples/make-demo /tmp/layerdiff-demo
//
// It writes base/ (a single .safetensors file) and tuned/ (the same
// model sharded into two files plus a *.safetensors.index.json), where
// "finetuning" touched layer 2 and the norm of layer 1, dropped a
// rotary buffer, and added a gate projection. Values come from a fixed
// xorshift PRNG, so every run produces byte-identical files.
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/JaydenCJ/layerdiff/internal/stwrite"
)

// rng is a self-contained xorshift64* generator: no dependency on
// math/rand sequence stability across Go releases.
type rng struct{ state uint64 }

func (r *rng) next() uint64 {
	r.state ^= r.state >> 12
	r.state ^= r.state << 25
	r.state ^= r.state >> 27
	return r.state * 0x2545F4914F6CDD1D
}

// uniform returns a float32 in [-scale, scale).
func (r *rng) uniform(scale float32) float32 {
	return (float32(r.next()>>40)/float32(1<<24)*2 - 1) * scale
}

func (r *rng) tensor(n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = r.uniform(scale)
	}
	return out
}

const (
	hidden = 16
	vocab  = 32
	ffn    = 64
	layers = 3
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: make-demo OUTDIR")
		os.Exit(2)
	}
	root := os.Args[1]
	if err := build(root); err != nil {
		fmt.Fprintln(os.Stderr, "make-demo:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s/base (single file) and %s/tuned (2 shards + index)\n", root, root)
}

func build(root string) error {
	base := makeBase(&rng{state: 20260712})
	tuned := makeTuned(base, &rng{state: 424242})

	baseDir := filepath.Join(root, "base")
	tunedDir := filepath.Join(root, "tuned")
	for _, d := range []string{baseDir, tunedDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
	}

	// base/: one classic single-file checkpoint.
	meta := map[string]string{"format": "pt", "producer": "layerdiff-demo"}
	if err := stwrite.WriteFile(filepath.Join(baseDir, "model.safetensors"), meta, base); err != nil {
		return err
	}

	// tuned/: the same model sharded in two, plus the index that maps
	// each tensor to its shard — the layout used for large models.
	cut := 0
	for i, t := range tuned {
		if t.Name == "model.layers.2.attn.wq.weight" {
			cut = i
			break
		}
	}
	shard1, shard2 := tuned[:cut], tuned[cut:]
	names := [2]string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
	if err := stwrite.WriteFile(filepath.Join(tunedDir, names[0]), nil, shard1); err != nil {
		return err
	}
	if err := stwrite.WriteFile(filepath.Join(tunedDir, names[1]), nil, shard2); err != nil {
		return err
	}

	weightMap := map[string]string{}
	total := 0
	for _, t := range shard1 {
		weightMap[t.Name] = names[0]
		total += len(t.Data)
	}
	for _, t := range shard2 {
		weightMap[t.Name] = names[1]
		total += len(t.Data)
	}
	index := map[string]any{
		"metadata":   map[string]any{"total_size": total},
		"weight_map": weightMap,
	}
	js, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(tunedDir, "model.safetensors.index.json"), append(js, '\n'), 0o644)
}

// makeBase builds the pre-finetune model.
func makeBase(r *rng) []stwrite.Tensor {
	var ts []stwrite.Tensor
	add := func(name string, shape []int64, scale float32) {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		ts = append(ts, stwrite.Tensor{
			Name: name, DType: "F32", Shape: shape,
			Data: stwrite.F32(r.tensor(n, scale)...),
		})
	}

	add("model.embed_tokens.weight", []int64{vocab, hidden}, 0.1)
	add("model.rotary.inv_freq", []int64{hidden / 2}, 1.0)
	for l := 0; l < layers; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		add(p+"attn.wq.weight", []int64{hidden, hidden}, 0.1)
		add(p+"attn.wk.weight", []int64{hidden, hidden}, 0.1)
		add(p+"attn.wo.weight", []int64{hidden, hidden}, 0.1)
		add(p+"mlp.up.weight", []int64{ffn, hidden}, 0.1)
		add(p+"mlp.down.weight", []int64{hidden, ffn}, 0.1)
		add(p+"norm.weight", []int64{hidden}, 0.02)
	}
	add("model.norm.weight", []int64{hidden}, 0.02)
	add("lm_head.weight", []int64{vocab, hidden}, 0.1)
	return ts
}

// makeTuned derives the post-finetune model: perturb layer 2's
// projections and layer 1's norm, drop the rotary buffer, add a gate.
func makeTuned(base []stwrite.Tensor, r *rng) []stwrite.Tensor {
	perturb := map[string]float32{
		"model.layers.2.attn.wq.weight": 0.03,
		"model.layers.2.attn.wk.weight": 0.02,
		"model.layers.2.mlp.up.weight":  0.01,
		"model.layers.1.norm.weight":    0.001,
		"lm_head.weight":                0.005,
	}
	var ts []stwrite.Tensor
	for _, t := range base {
		if t.Name == "model.rotary.inv_freq" {
			continue // finetuning pipeline stopped persisting the buffer
		}
		data := append([]byte(nil), t.Data...)
		if scale, hit := perturb[t.Name]; hit {
			vals := decodeF32(data)
			for i := range vals {
				vals[i] += r.uniform(scale)
			}
			data = stwrite.F32(vals...)
		}
		ts = append(ts, stwrite.Tensor{Name: t.Name, DType: t.DType, Shape: t.Shape, Data: data})

		// The tuned model gained a gate projection in layer 2.
		if t.Name == "model.layers.2.mlp.down.weight" {
			ts = append(ts, stwrite.Tensor{
				Name: "model.layers.2.mlp.gate.weight", DType: "F32",
				Shape: []int64{ffn, hidden},
				Data:  stwrite.F32(r.tensor(ffn*hidden, 0.1)...),
			})
		}
	}
	return ts
}

func decodeF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}
