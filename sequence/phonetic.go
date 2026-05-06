package sequence

// Soundex returns the Russell-O'Dell Soundex phonetic code for a name string,
// per the modern (Knuth Volume 3 / SSA-refined) algorithm including the H/W
// transparency rule.
//
// The result is a 4-character code consisting of one uppercase letter (the
// first letter of s, preserved) followed by three digits. Names that hash to
// the same Soundex code are likely homophones in English (e.g. "Robert" and
// "Rupert" both return "R163"; "Ashcraft" returns "A261" via the H/W
// transparency rule).
//
// Empty input or input whose first rune is not an ASCII letter returns the
// empty string. Non-alphabetic runes elsewhere in the input are skipped.
// Input is case-insensitive (the first letter is uppercased in the output).
//
// Algorithm:
//   1. Retain the first letter (uppercased).
//   2. Encode each subsequent letter as a digit:
//        B,F,P,V         -> 1
//        C,G,J,K,Q,S,X,Z -> 2
//        D,T             -> 3
//        L               -> 4
//        M,N             -> 5
//        R               -> 6
//        A,E,I,O,U,Y     -> 0  (vowel; treated as a separator)
//        H,W             -> transparent (skip, but do NOT reset the running class)
//   3. If two same-class consonants are adjacent — INCLUDING when separated
//      only by H or W — keep only the first.  If separated by a vowel
//      (A/E/I/O/U/Y), keep both.  This is the modern Soundex H/W rule.
//   4. The first letter's class is also part of the running class, so a
//      same-class consonant immediately after the first letter is dropped
//      (e.g. "Pfister" -> "P236", not "P1236", because F shares P's class 1).
//   5. Truncate or zero-pad to 4 characters.
//
// References:
//   - Russell, R. C. (1918). U.S. Patent 1,261,167.
//   - Knuth, D. E. (1973). The Art of Computer Programming, Volume 3:
//     Sorting and Searching, §6.4 (1st ed).
//   - U.S. Social Security Administration (1980). The Soundex Indexing
//     System (NARA reference).
func Soundex(s string) string {
	if len(s) == 0 {
		return ""
	}
	runes := []rune(s)
	if len(runes) == 0 {
		return ""
	}

	first := runes[0]
	if first >= 'a' && first <= 'z' {
		first -= 'a' - 'A'
	}
	if first < 'A' || first > 'Z' {
		return ""
	}

	out := []byte{byte(first)}
	prevCode := soundexCode(byte(first))

	for i := 1; i < len(runes); i++ {
		r := runes[i]
		if r >= 'a' && r <= 'z' {
			r -= 'a' - 'A'
		}
		if r < 'A' || r > 'Z' {
			continue
		}
		if r == 'H' || r == 'W' {
			// Transparent — skip without resetting prevCode so adjacent
			// same-class consonants on either side still collapse.
			continue
		}
		c := soundexCode(byte(r))
		if c == '0' {
			// Vowel: separator. Reset running class so subsequent same-
			// class consonants on either side of the vowel are both kept.
			prevCode = '0'
			continue
		}
		if c == prevCode {
			// Adjacent same-class (after H/W transparency): collapse.
			continue
		}
		out = append(out, c)
		prevCode = c
		if len(out) >= 4 {
			break
		}
	}

	for len(out) < 4 {
		out = append(out, '0')
	}
	return string(out)
}

// soundexCode returns the Soundex digit class of an uppercase ASCII letter,
// or '0' for vowels (A,E,I,O,U,Y) and the H/W transparent letters (which the
// caller handles before invocation; this function is consulted only as a
// fallback for first-letter prevCode initialisation).
func soundexCode(r byte) byte {
	switch r {
	case 'B', 'F', 'P', 'V':
		return '1'
	case 'C', 'G', 'J', 'K', 'Q', 'S', 'X', 'Z':
		return '2'
	case 'D', 'T':
		return '3'
	case 'L':
		return '4'
	case 'M', 'N':
		return '5'
	case 'R':
		return '6'
	}
	return '0'
}
