// Tests for the fixture writer: what it emits must round-trip through
// the real parser, because every other test in the repository builds its
// checkpoints with it.
package stwrite

import (
	"path/filepath"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/safetensors"
)

func TestEncodeRoundTripsThroughParser(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rt.safetensors")
	err := WriteFile(path, map[string]string{"k": "v"}, []Tensor{
		{Name: "a", DType: "F32", Shape: []int64{2, 2}, Data: F32(1, 2, 3, 4)},
		{Name: "b", DType: "I64", Shape: []int64{2}, Data: I64(-1, 9)},
	})
	if err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	sf, err := safetensors.Open(path)
	if err != nil {
		t.Fatalf("parser rejected writer output: %v", err)
	}
	if len(sf.Tensors) != 2 || sf.Metadata["k"] != "v" {
		t.Fatalf("round trip lost content: %+v", sf)
	}
	if sf.Tensors[0].Name != "a" || sf.Tensors[0].Elems() != 4 || sf.Tensors[0].Bytes() != 16 {
		t.Fatalf("tensor a mangled: %+v", sf.Tensors[0])
	}
}

func TestEncodeHeaderAlignedTo8(t *testing.T) {
	img, err := Encode(nil, []Tensor{{Name: "w", DType: "U8", Shape: []int64{1}, Data: []byte{7}}})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	headerLen := int64(img[0]) | int64(img[1])<<8 // header < 256 bytes here
	if headerLen%8 != 0 {
		t.Fatalf("header length %d not 8-byte aligned", headerLen)
	}
}

func TestEncodeRejectsInvalidTensors(t *testing.T) {
	if _, err := Encode(nil, []Tensor{{Name: "w", DType: "F32", Shape: []int64{3}, Data: F32(1, 2)}}); err == nil {
		t.Fatal("shape/data size mismatch must be rejected")
	}
	if _, err := Encode(nil, []Tensor{
		{Name: "w", DType: "U8", Shape: []int64{1}, Data: []byte{1}},
		{Name: "w", DType: "U8", Shape: []int64{1}, Data: []byte{2}},
	}); err == nil {
		t.Fatal("duplicate tensor names must be rejected")
	}
}

func TestPackHelpers(t *testing.T) {
	if b := F32(1); len(b) != 4 || b[3] != 0x3f || b[2] != 0x80 {
		t.Errorf("F32(1) = % x", b)
	}
	if b := F64(1); len(b) != 8 || b[7] != 0x3f || b[6] != 0xf0 {
		t.Errorf("F64(1) = % x", b)
	}
	if b := U16(0x1234); len(b) != 2 || b[0] != 0x34 || b[1] != 0x12 {
		t.Errorf("U16 = % x", b)
	}
	if b := I64(-1); len(b) != 8 || b[0] != 0xff || b[7] != 0xff {
		t.Errorf("I64(-1) = % x", b)
	}
	if b := I8(-2); len(b) != 1 || b[0] != 0xfe {
		t.Errorf("I8(-2) = % x", b)
	}
}
