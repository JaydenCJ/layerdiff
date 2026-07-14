// Tests for hash manifests: building, byte-deterministic writing,
// loading, and the sniffing that separates a manifest from a sharded
// index when both arrive as ".json".
package manifest

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/stwrite"
)

func fixtureCheckpoint(t *testing.T) *checkpoint.Checkpoint {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors")
	err := stwrite.WriteFile(path, nil, []stwrite.Tensor{
		{Name: "a.weight", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(1, 2)},
		{Name: "b.bias", DType: "I8", Shape: []int64{1}, Data: stwrite.I8(-3)},
	})
	if err != nil {
		t.Fatal(err)
	}
	cp, err := checkpoint.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return cp
}

func TestBuildRecordsEveryTensor(t *testing.T) {
	m, err := Build(fixtureCheckpoint(t))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if m.Schema != Schema || m.Tool != "layerdiff" || len(m.Tensors) != 2 {
		t.Fatalf("manifest header wrong: %+v", m)
	}
	e := m.Tensors["a.weight"]
	if e.DType != "F32" || !reflect.DeepEqual(e.Shape, []int64{2}) || e.Bytes != 8 || len(e.SHA256) != 64 {
		t.Fatalf("entry wrong: %+v", e)
	}
}

func TestWriteThenLoadRoundTrips(t *testing.T) {
	m, err := Build(fixtureCheckpoint(t))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "m.json")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Write(f); err != nil {
		t.Fatal(err)
	}
	f.Close()

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !reflect.DeepEqual(loaded, m) {
		t.Fatalf("round trip diverged:\n%+v\n%+v", loaded, m)
	}
}

func TestWriteIsByteDeterministic(t *testing.T) {
	m, err := Build(fixtureCheckpoint(t))
	if err != nil {
		t.Fatal(err)
	}
	var one, two bytes.Buffer
	if err := m.Write(&one); err != nil {
		t.Fatal(err)
	}
	if err := m.Write(&two); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(one.Bytes(), two.Bytes()) {
		t.Fatal("two writes of the same manifest differ")
	}
}

func TestSniffAcceptsManifestRejectsIndex(t *testing.T) {
	dir := t.TempDir()
	man := filepath.Join(dir, "m.json")
	idx := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(man, []byte(`{"layerdiff_manifest":1,"tensors":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(idx, []byte(`{"weight_map":{"w":"s.safetensors"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if !Sniff(man) {
		t.Fatal("manifest not recognized")
	}
	if Sniff(idx) {
		t.Fatal("sharded index misrecognized as manifest")
	}
	if Sniff(filepath.Join(dir, "missing.json")) {
		t.Fatal("missing file sniffed as manifest")
	}
}

func TestLoadRejectsBadManifests(t *testing.T) {
	cases := map[string]string{
		"newer schema":         `{"layerdiff_manifest":99,"tensors":{}}`,
		"missing marker":       `{"tensors":{}}`,
		"entry without sha256": `{"layerdiff_manifest":1,"tensors":{"w":{"dtype":"F32","shape":[1],"bytes":4}}}`,
	}
	for name, doc := range cases {
		path := filepath.Join(t.TempDir(), "m.json")
		if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(path); err == nil {
			t.Errorf("%s: bad manifest accepted", name)
		}
	}
}
