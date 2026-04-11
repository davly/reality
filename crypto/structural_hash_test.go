package crypto

import "testing"

// =========================================================================
// SituationHashWithStructure tests
// =========================================================================

func TestSituationHashWithStructure_Deterministic(t *testing.T) {
	content := []byte("temperature=22.5&humidity=45")
	structure := []byte{2, 11, 8} // 2 fields, key lengths 11 and 8
	h1 := SituationHashWithStructure(content, structure)
	h2 := SituationHashWithStructure(content, structure)
	if h1 != h2 {
		t.Errorf("SituationHashWithStructure not deterministic: %d != %d", h1, h2)
	}
}

func TestSituationHashWithStructure_DifferentStructureDifferentHash(t *testing.T) {
	content := []byte("data=42")
	struct1 := []byte{1, 4}    // 1 field, key length 4
	struct2 := []byte{2, 2, 2} // 2 fields, key lengths 2 and 2
	h1 := SituationHashWithStructure(content, struct1)
	h2 := SituationHashWithStructure(content, struct2)
	if h1 == h2 {
		t.Error("same content with different structures should produce different hashes")
	}
}

func TestSituationHashWithStructure_DifferentContentDifferentHash(t *testing.T) {
	structure := []byte{1, 5}
	h1 := SituationHashWithStructure([]byte("hello"), structure)
	h2 := SituationHashWithStructure([]byte("world"), structure)
	if h1 == h2 {
		t.Error("different content with same structure should produce different hashes")
	}
}

func TestSituationHashWithStructure_EmptyInputs(t *testing.T) {
	// Empty content and structure should not panic.
	h := SituationHashWithStructure(nil, nil)
	// Just verify it returns something deterministic.
	h2 := SituationHashWithStructure(nil, nil)
	if h != h2 {
		t.Errorf("empty inputs not deterministic: %d != %d", h, h2)
	}
}

// =========================================================================
// StructuralDescriptor tests
// =========================================================================

func TestStructuralDescriptor_Basic(t *testing.T) {
	keys := []string{"name", "age", "score"}
	desc := StructuralDescriptor(keys)
	if len(desc) != 4 {
		t.Fatalf("descriptor length = %d, want 4", len(desc))
	}
	if desc[0] != 3 {
		t.Errorf("field count = %d, want 3", desc[0])
	}
	if desc[1] != 4 { // "name" = 4
		t.Errorf("key[0] length = %d, want 4", desc[1])
	}
	if desc[2] != 3 { // "age" = 3
		t.Errorf("key[1] length = %d, want 3", desc[2])
	}
	if desc[3] != 5 { // "score" = 5
		t.Errorf("key[2] length = %d, want 5", desc[3])
	}
}

func TestStructuralDescriptor_Empty(t *testing.T) {
	desc := StructuralDescriptor(nil)
	if len(desc) != 1 || desc[0] != 0 {
		t.Errorf("descriptor for nil keys = %v, want [0]", desc)
	}
}

func TestStructuralDescriptor_OrderMatters(t *testing.T) {
	desc1 := StructuralDescriptor([]string{"ab", "cde"})
	desc2 := StructuralDescriptor([]string{"cde", "ab"})
	// Same key set but different order should produce different descriptors.
	if len(desc1) != len(desc2) {
		t.Fatal("descriptor lengths differ for same-size key sets")
	}
	different := false
	for i := range desc1 {
		if desc1[i] != desc2[i] {
			different = true
			break
		}
	}
	if !different {
		t.Error("different key orderings produced identical descriptors")
	}
}

func TestSituationHashWithStructure_IntegratedWithDescriptor(t *testing.T) {
	// End-to-end: build descriptor, then hash.
	content := []byte("temperature=22.5&pressure=1013")
	desc := StructuralDescriptor([]string{"temperature", "pressure"})
	h := SituationHashWithStructure(content, desc)
	// Verify it differs from content-only hash.
	contentOnly := FNV1a64(content)
	if h == contentOnly {
		t.Error("structural hash should differ from content-only FNV1a64")
	}
}
