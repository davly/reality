package sequence

import "testing"

// TestSoundex_KnuthCanonicalCases pins the implementation against the
// canonical Knuth Volume 3 examples that the modern Soundex algorithm was
// chosen to match.
func TestSoundex_KnuthCanonicalCases(t *testing.T) {
	cases := []struct {
		name string
		want string
	}{
		// Russell-O'Dell originals.
		{"Robert", "R163"},
		{"Rupert", "R163"},
		{"Rubin", "R150"},

		// Knuth's H/W rule examples — Ashcraft tests the transparency
		// (S and C are both class 2; separated only by H, so C is dropped).
		{"Ashcraft", "A261"},
		{"Ashcroft", "A261"},

		// Tymczak: M, C, Z all distinct classes; separator vowels reset.
		{"Tymczak", "T522"},

		// Pfister: F shares P's class so F is dropped at the boundary.
		{"Pfister", "P236"},

		// Honeyman: 3 N/M consonants separated by vowels — all 3 retained.
		{"Honeyman", "H555"},

		// Lloyd: double L collapses; D separate.
		{"Lloyd", "L300"},

		// Lee: short input pads with zeros.
		{"Lee", "L000"},

		// Single character: pads to L000-style.
		{"L", "L000"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Soundex(tc.name)
			if got != tc.want {
				t.Errorf("Soundex(%q) = %q, want %q", tc.name, got, tc.want)
			}
		})
	}
}

// TestSoundex_CaseInsensitive verifies that uppercase, lowercase, and mixed
// case all produce the same code.
func TestSoundex_CaseInsensitive(t *testing.T) {
	for _, s := range []string{"robert", "ROBERT", "Robert", "RoBeRt"} {
		if got := Soundex(s); got != "R163" {
			t.Errorf("Soundex(%q) = %q, want R163", s, got)
		}
	}
}

// TestSoundex_NonAlphaInputs tests the boundary behaviour for empty,
// non-alphabetic-leading, and embedded-non-alpha inputs.
func TestSoundex_NonAlphaInputs(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"123", ""},
		{"  Robert", ""},
		{"Robert!", "R163"},
		{"Robert-Smith", "R163"}, // hyphen ignored; truncates after 4 chars on 'S' boundary
		{"O'Brien", "O165"},      // apostrophe ignored
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got := Soundex(tc.in)
			if got != tc.want {
				t.Errorf("Soundex(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestSoundex_SameCodeForHomophones exercises the homophone-clustering
// property that Soundex was designed for: similar-sounding name pairs should
// produce identical codes. Note Soundex preserves the FIRST LETTER literally,
// so true homophones differing in their leading letter (e.g. "Catherine" /
// "Kathryn") do NOT cluster — this is a well-known Soundex limitation.
func TestSoundex_SameCodeForHomophones(t *testing.T) {
	pairs := [][2]string{
		{"Smith", "Smyth"},
		{"Robert", "Rupert"},
		{"Ashcraft", "Ashcroft"},
	}
	for _, p := range pairs {
		t.Run(p[0]+"_vs_"+p[1], func(t *testing.T) {
			a, b := Soundex(p[0]), Soundex(p[1])
			if a != b {
				t.Errorf("Soundex(%q)=%q != Soundex(%q)=%q (expected homophone match)",
					p[0], a, p[1], b)
			}
		})
	}
}

// TestSoundex_DifferentCodeForDistinctNames verifies that structurally
// different names get distinct codes (the converse of the homophone test).
func TestSoundex_DifferentCodeForDistinctNames(t *testing.T) {
	pairs := [][2]string{
		{"Robert", "Smith"},
		{"Tymczak", "Honeyman"},
	}
	for _, p := range pairs {
		t.Run(p[0]+"_vs_"+p[1], func(t *testing.T) {
			a, b := Soundex(p[0]), Soundex(p[1])
			if a == b {
				t.Errorf("Soundex(%q) == Soundex(%q) == %q (expected distinct codes)",
					p[0], p[1], a)
			}
		})
	}
}
