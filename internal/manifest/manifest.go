// Package manifest snapshots a checkpoint's identity — every tensor's
// dtype, shape, byte size, and SHA-256 — as a small JSON document. A
// manifest can stand in for either side of a diff, so "did anything
// change since the run I archived?" is answerable without keeping the
// old weights around.
package manifest

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/scan"
	"github.com/JaydenCJ/layerdiff/internal/version"
)

// Schema is the manifest schema version this build reads and writes.
const Schema = 1

// Entry records one tensor's identity.
type Entry struct {
	DType  string  `json:"dtype"`
	Shape  []int64 `json:"shape"`
	Bytes  int64   `json:"bytes"`
	SHA256 string  `json:"sha256"`
}

// Manifest is the on-disk document. The layerdiff_manifest field doubles
// as the sniffing marker that distinguishes a manifest from a sharded
// index when a .json path is passed to diff.
type Manifest struct {
	Schema  int              `json:"layerdiff_manifest"`
	Tool    string           `json:"tool"`
	Version string           `json:"version"`
	Source  string           `json:"source"`
	Tensors map[string]Entry `json:"tensors"`
}

// Build scans every tensor of the checkpoint (hash-only, constant
// memory) and returns its manifest.
func Build(cp *checkpoint.Checkpoint) (*Manifest, error) {
	m := &Manifest{
		Schema:  Schema,
		Tool:    "layerdiff",
		Version: version.Version,
		Source:  filepath.Base(cp.Path),
		Tensors: make(map[string]Entry, cp.Len()),
	}
	for _, name := range cp.Names() {
		t, _ := cp.Tensor(name)
		res, err := scan.HashOnly(t)
		if err != nil {
			return nil, err
		}
		shape := t.Shape
		if shape == nil {
			shape = []int64{}
		}
		m.Tensors[name] = Entry{DType: t.DType, Shape: shape, Bytes: res.Bytes, SHA256: res.SHA256}
	}
	return m, nil
}

// Write emits the manifest as indented JSON. encoding/json sorts map
// keys, so output is deterministic byte for byte.
func (m *Manifest) Write(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(m)
}

// Load reads and validates a manifest file.
func Load(path string) (*Manifest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("%s: not a layerdiff manifest: %w", path, err)
	}
	if m.Schema == 0 {
		return nil, fmt.Errorf("%s: not a layerdiff manifest: missing layerdiff_manifest field", path)
	}
	if m.Schema > Schema {
		return nil, fmt.Errorf("%s: manifest schema %d is newer than this build understands (%d)", path, m.Schema, Schema)
	}
	for name, e := range m.Tensors {
		if e.DType == "" || e.SHA256 == "" {
			return nil, fmt.Errorf("%s: tensor %q: entry missing dtype or sha256", path, name)
		}
	}
	return &m, nil
}

// Sniff reports whether the file at path parses as a layerdiff manifest.
// It never returns an error: any unreadable or foreign file is simply
// not a manifest.
func Sniff(path string) bool {
	raw, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var probe struct {
		Schema int `json:"layerdiff_manifest"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	return probe.Schema > 0
}
