// Tests for checkpoint resolution: single files, directories of loose
// shards, sharded indexes, and the failure modes of each (duplicates,
// missing shards, hostile index paths).
package checkpoint

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/stwrite"
)

func writeST(t *testing.T, path string, meta map[string]string, tensors []stwrite.Tensor) {
	t.Helper()
	if err := stwrite.WriteFile(path, meta, tensors); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func writeIndex(t *testing.T, path string, weightMap map[string]string, meta map[string]any) {
	t.Helper()
	doc := map[string]any{"weight_map": weightMap}
	if meta != nil {
		doc["metadata"] = meta
	}
	js, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, js, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestOpenSingleFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	writeST(t, path, map[string]string{"format": "pt"}, []stwrite.Tensor{
		{Name: "model.layers.10.w", DType: "F32", Shape: []int64{1}, Data: stwrite.F32(1)},
		{Name: "model.layers.2.w", DType: "F32", Shape: []int64{1}, Data: stwrite.F32(2)},
	})
	cp, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if cp.Kind != KindFile || cp.Len() != 2 || cp.Metadata["format"] != "pt" {
		t.Fatalf("unexpected checkpoint: kind=%s len=%d meta=%v", cp.Kind, cp.Len(), cp.Metadata)
	}
	// Natural order: layer 2 before layer 10.
	want := []string{"model.layers.2.w", "model.layers.10.w"}
	if !reflect.DeepEqual(cp.Names(), want) {
		t.Fatalf("names = %v, want %v", cp.Names(), want)
	}
}

func TestTensorReaderStreamsExactBytes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	writeST(t, path, nil, []stwrite.Tensor{
		{Name: "a", DType: "U8", Shape: []int64{3}, Data: []byte{1, 2, 3}},
		{Name: "b", DType: "U8", Shape: []int64{2}, Data: []byte{9, 8}},
	})
	cp, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tb, ok := cp.Tensor("b")
	if !ok {
		t.Fatal("tensor b missing")
	}
	r, err := tb.Reader()
	if err != nil {
		t.Fatalf("Reader: %v", err)
	}
	defer r.Close()
	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	// The reader must yield exactly b's bytes — not a's, not the header.
	if !reflect.DeepEqual(got, []byte{9, 8}) {
		t.Fatalf("read % x, want 09 08", got)
	}
}

func TestOpenDirectoryMergesLooseShards(t *testing.T) {
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "part-a.safetensors"), nil, []stwrite.Tensor{
		{Name: "x", DType: "U8", Shape: []int64{1}, Data: []byte{1}},
	})
	writeST(t, filepath.Join(dir, "part-b.safetensors"), nil, []stwrite.Tensor{
		{Name: "y", DType: "U8", Shape: []int64{1}, Data: []byte{2}},
	})
	cp, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if cp.Kind != KindDirectory || cp.Len() != 2 || len(cp.Shards) != 2 {
		t.Fatalf("directory merge failed: kind=%s len=%d shards=%v", cp.Kind, cp.Len(), cp.Shards)
	}

	// A directory holding exactly one file degrades to a plain file open.
	single := t.TempDir()
	writeST(t, filepath.Join(single, "only.safetensors"), nil, []stwrite.Tensor{
		{Name: "x", DType: "U8", Shape: []int64{1}, Data: []byte{1}},
	})
	cp, err = Open(single)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if cp.Kind != KindFile {
		t.Fatalf("single-file directory should report %q, got %q", KindFile, cp.Kind)
	}
}

func TestOpenDirectoryDuplicateTensorRejected(t *testing.T) {
	dir := t.TempDir()
	tensor := []stwrite.Tensor{{Name: "same", DType: "U8", Shape: []int64{1}, Data: []byte{1}}}
	writeST(t, filepath.Join(dir, "a.safetensors"), nil, tensor)
	writeST(t, filepath.Join(dir, "b.safetensors"), nil, tensor)
	if _, err := Open(dir); err == nil || !strings.Contains(err.Error(), "appears in both") {
		t.Fatalf("want duplicate error, got %v", err)
	}
}

func TestOpenShardedIndex(t *testing.T) {
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), nil, []stwrite.Tensor{
		{Name: "model.layers.0.w", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(1, 2)},
	})
	writeST(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), nil, []stwrite.Tensor{
		{Name: "model.layers.1.w", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(3, 4)},
	})
	idx := filepath.Join(dir, "model.safetensors.index.json")
	writeIndex(t, idx, map[string]string{
		"model.layers.0.w": "model-00001-of-00002.safetensors",
		"model.layers.1.w": "model-00002-of-00002.safetensors",
	}, map[string]any{"total_size": 16})

	cp, err := Open(idx)
	if err != nil {
		t.Fatalf("Open index: %v", err)
	}
	if cp.Kind != KindSharded || cp.Len() != 2 || len(cp.Shards) != 2 {
		t.Fatalf("sharded open failed: kind=%s len=%d shards=%v", cp.Kind, cp.Len(), cp.Shards)
	}
	if cp.Metadata["total_size"] != "16" {
		t.Fatalf("index metadata not stringified: %v", cp.Metadata)
	}
	if cp.DataBytes() != 16 {
		t.Fatalf("DataBytes = %d, want 16", cp.DataBytes())
	}
}

func TestOpenDirectoryPrefersIndex(t *testing.T) {
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "model-00001-of-00001.safetensors"), nil, []stwrite.Tensor{
		{Name: "w", DType: "U8", Shape: []int64{1}, Data: []byte{1}},
		{Name: "not.mapped", DType: "U8", Shape: []int64{1}, Data: []byte{2}},
	})
	writeIndex(t, filepath.Join(dir, "model.safetensors.index.json"),
		map[string]string{"w": "model-00001-of-00001.safetensors"}, nil)

	cp, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if cp.Kind != KindSharded {
		t.Fatalf("directory with index must resolve as sharded, got %q", cp.Kind)
	}
	if cp.Path != dir {
		t.Fatalf("report label must stay the user's path, got %q", cp.Path)
	}
	// Only weight_map entries belong to the checkpoint.
	if cp.Len() != 1 {
		t.Fatalf("unmapped tensors must be ignored, got %v", cp.Names())
	}
}

func TestOpenIndexRejectsMalformedIndexes(t *testing.T) {
	// A tensor mapped to a shard that does not contain it.
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "s.safetensors"), nil, []stwrite.Tensor{
		{Name: "present", DType: "U8", Shape: []int64{1}, Data: []byte{1}},
	})
	idx := filepath.Join(dir, "model.safetensors.index.json")
	writeIndex(t, idx, map[string]string{"ghost": "s.safetensors"}, nil)
	if _, err := Open(idx); err == nil || !strings.Contains(err.Error(), "does not contain") {
		t.Fatalf("want missing-tensor error, got %v", err)
	}

	// A weight map that tries to escape the index's directory.
	dir2 := t.TempDir()
	idx2 := filepath.Join(dir2, "model.safetensors.index.json")
	writeIndex(t, idx2, map[string]string{"w": "../outside.safetensors"}, nil)
	if _, err := Open(idx2); err == nil || !strings.Contains(err.Error(), "invalid shard name") {
		t.Fatalf("want invalid-shard error, got %v", err)
	}

	// No weight_map at all: this is some other kind of JSON.
	dir3 := t.TempDir()
	idx3 := filepath.Join(dir3, "model.safetensors.index.json")
	if err := os.WriteFile(idx3, []byte(`{"metadata":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(idx3); err == nil || !strings.Contains(err.Error(), "weight_map") {
		t.Fatalf("want weight_map error, got %v", err)
	}
}

func TestOpenRejectsMissingAndEmptyPaths(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "nope")); err == nil {
		t.Fatal("missing path must error")
	}
	if _, err := Open(t.TempDir()); err == nil {
		t.Fatal("directory without checkpoints must be rejected")
	}
}
