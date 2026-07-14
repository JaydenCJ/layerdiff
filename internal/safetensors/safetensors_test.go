// Tests for safetensors header parsing. Corrupt-file cases are crafted
// byte by byte because the whole point is rejecting inputs the writer
// could never produce.
package safetensors

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/stwrite"
)

// writeFixture writes a well-formed file via stwrite and returns its path.
func writeFixture(t *testing.T, meta map[string]string, tensors []stwrite.Tensor) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := stwrite.WriteFile(path, meta, tensors); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

// writeRaw writes an arbitrary byte image as a .safetensors file.
func writeRaw(t *testing.T, b []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "corrupt.safetensors")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write raw: %v", err)
	}
	return path
}

// rawWithHeader assembles length-prefix + header JSON + data.
func rawWithHeader(header string, data []byte) []byte {
	out := make([]byte, 8, 8+len(header)+len(data))
	binary.LittleEndian.PutUint64(out, uint64(len(header)))
	out = append(out, header...)
	return append(out, data...)
}

func TestOpenParsesTensorsSortedByName(t *testing.T) {
	path := writeFixture(t, nil, []stwrite.Tensor{
		{Name: "b.weight", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(1, 2)},
		{Name: "a.weight", DType: "F32", Shape: []int64{1}, Data: stwrite.F32(3)},
	})
	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(sf.Tensors) != 2 || sf.Tensors[0].Name != "a.weight" || sf.Tensors[1].Name != "b.weight" {
		t.Fatalf("tensors not sorted by name: %+v", sf.Tensors)
	}
	// b.weight was laid out first, so its data-relative offset is 0;
	// absolute offsets must account for the header.
	b := sf.Tensors[1]
	if b.Begin != sf.DataStart || b.End != sf.DataStart+8 {
		t.Fatalf("absolute offsets wrong: begin=%d end=%d dataStart=%d", b.Begin, b.End, sf.DataStart)
	}
}

func TestOpenReadsMetadata(t *testing.T) {
	path := writeFixture(t, map[string]string{"format": "pt", "producer": "unit-test"},
		[]stwrite.Tensor{{Name: "w", DType: "F32", Shape: []int64{1}, Data: stwrite.F32(0)}})
	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if sf.Metadata["format"] != "pt" || sf.Metadata["producer"] != "unit-test" {
		t.Fatalf("metadata not parsed: %+v", sf.Metadata)
	}
}

func TestScalarAndEmptyShapes(t *testing.T) {
	path := writeFixture(t, nil, []stwrite.Tensor{
		{Name: "scalar", DType: "F32", Shape: nil, Data: stwrite.F32(7)},
		{Name: "empty", DType: "F32", Shape: []int64{0, 4}, Data: nil},
	})
	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	byName := map[string]Tensor{}
	for _, tt := range sf.Tensors {
		byName[tt.Name] = tt
	}
	if got := byName["scalar"].Elems(); got != 1 {
		t.Errorf("rank-0 tensor must have 1 element, got %d", got)
	}
	if got := byName["empty"].Elems(); got != 0 {
		t.Errorf("zero-dim tensor must have 0 elements, got %d", got)
	}
	if got := byName["empty"].Bytes(); got != 0 {
		t.Errorf("zero-dim tensor must span 0 bytes, got %d", got)
	}
}

func TestHeaderPaddingWithSpacesAccepted(t *testing.T) {
	// The reference writer pads headers to 8-byte alignment with spaces.
	header := `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}    `
	path := writeRaw(t, rawWithHeader(header, stwrite.F32(1)))
	if _, err := Open(path); err != nil {
		t.Fatalf("padded header must parse: %v", err)
	}
}

func TestCorruptFilesRejected(t *testing.T) {
	hugeLen := make([]byte, 8)
	binary.LittleEndian.PutUint64(hugeLen, uint64(MaxHeaderSize)+1)
	overrunLen := make([]byte, 8)
	binary.LittleEndian.PutUint64(overrunLen, 4096) // claims 4 KiB header, file has none

	cases := []struct {
		name    string
		image   []byte
		wantErr string // substring of the error, "" = any error
	}{
		{"file shorter than length prefix", []byte{1, 2, 3}, "not a safetensors file"},
		{"header length overruns file", overrunLen, "overruns"},
		{"header length exceeds cap", hugeLen, "cap"},
		{"malformed header JSON", rawWithHeader(`{"w": nope}`, nil), ""},
		// Shape [3] × F32 needs 12 bytes but the offsets span 8.
		{"shape/byte-length mismatch",
			rawWithHeader(`{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}`, make([]byte, 8)),
			"needs 12 bytes"},
		{"negative dimension",
			rawWithHeader(`{"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}`, stwrite.F32(1)),
			"negative dimension"},
		{"offsets overrun data region",
			rawWithHeader(`{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`, make([]byte, 8)),
			"overruns"},
		{"backwards offsets",
			rawWithHeader(`{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}`, make([]byte, 8)),
			"invalid data_offsets"},
		{"non-string metadata values",
			rawWithHeader(`{"__metadata__":{"nested":{"x":1}}}`, nil),
			"__metadata__"},
	}
	for _, c := range cases {
		_, err := Open(writeRaw(t, c.image))
		if err == nil {
			t.Errorf("%s: corrupt file accepted", c.name)
			continue
		}
		if c.wantErr != "" && !strings.Contains(err.Error(), c.wantErr) {
			t.Errorf("%s: error %q does not mention %q", c.name, err, c.wantErr)
		}
	}
}

func TestUnknownDTypeAcceptedAsOpaque(t *testing.T) {
	// A future dtype must not break parsing: size validation is skipped,
	// byte range is still honored.
	header := `{"w":{"dtype":"F4_SOMEDAY","shape":[5],"data_offsets":[0,3]}}`
	path := writeRaw(t, rawWithHeader(header, []byte{1, 2, 3}))
	sf, err := Open(path)
	if err != nil {
		t.Fatalf("opaque dtype must parse: %v", err)
	}
	if sf.Tensors[0].Bytes() != 3 {
		t.Fatalf("opaque tensor byte span wrong: %d", sf.Tensors[0].Bytes())
	}
}
