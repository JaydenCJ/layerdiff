// Sharded-index support: *.safetensors.index.json maps each tensor name
// to the shard file that stores it. Only mapped tensors are part of the
// checkpoint, matching the semantics loaders apply to the weight map.
package checkpoint

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/JaydenCJ/layerdiff/internal/layer"
	"github.com/JaydenCJ/layerdiff/internal/safetensors"
)

type indexFile struct {
	Metadata  map[string]any    `json:"metadata"`
	WeightMap map[string]string `json:"weight_map"`
}

func openIndex(path string) (*Checkpoint, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var idx indexFile
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("%s: not a sharded index: %w", path, err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("%s: not a sharded index: missing or empty weight_map", path)
	}

	dir := filepath.Dir(path)
	byShard := make(map[string][]string)
	for tensor, shard := range idx.WeightMap {
		// Shard references must be plain file names: an index that points
		// outside its own directory is malformed (or hostile).
		if shard == "" || shard != filepath.Base(shard) || strings.HasPrefix(shard, ".") {
			return nil, fmt.Errorf("%s: tensor %q maps to invalid shard name %q", path, tensor, shard)
		}
		byShard[shard] = append(byShard[shard], tensor)
	}
	shards := make([]string, 0, len(byShard))
	for s := range byShard {
		shards = append(shards, s)
	}
	sort.Strings(shards)

	cp := &Checkpoint{
		Path:     path,
		Kind:     KindSharded,
		Shards:   shards,
		Metadata: stringifyMetadata(idx.Metadata),
		byName:   make(map[string]*Tensor, len(idx.WeightMap)),
	}
	for _, shard := range shards {
		sf, err := safetensors.Open(filepath.Join(dir, shard))
		if err != nil {
			return nil, fmt.Errorf("shard %s: %w", shard, err)
		}
		present := make(map[string]safetensors.Tensor, len(sf.Tensors))
		for _, t := range sf.Tensors {
			present[t.Name] = t
		}
		names := byShard[shard]
		layer.SortNames(names)
		for _, name := range names {
			t, ok := present[name]
			if !ok {
				return nil, fmt.Errorf("index maps tensor %q to %s, but the shard does not contain it", name, shard)
			}
			cp.byName[name] = &Tensor{
				Name:  t.Name,
				DType: t.DType,
				Shape: t.Shape,
				Bytes: t.Bytes(),
				Elems: t.Elems(),
				path:  sf.Path,
				begin: t.Begin,
				end:   t.End,
			}
		}
	}
	cp.finish()
	return cp, nil
}

// stringifyMetadata flattens the index "metadata" object (which mixes
// strings and numbers, e.g. total_size) into displayable strings.
func stringifyMetadata(m map[string]any) map[string]string {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		switch t := v.(type) {
		case string:
			out[k] = t
		case float64:
			out[k] = strconv.FormatFloat(t, 'f', -1, 64)
		case bool:
			out[k] = strconv.FormatBool(t)
		default:
			b, err := json.Marshal(v)
			if err != nil {
				continue
			}
			out[k] = string(b)
		}
	}
	return out
}
