package constants

import (
	"testing"

	"github.com/davly/reality/testutil"
)

// physicsConstantMap maps constant names (as used in golden-file inputs)
// to their actual values. This is the bridge between JSON test vectors
// and Go constants.
var physicsConstantMap = map[string]float64{
	"SpeedOfLight":       SpeedOfLight,
	"Planck":             Planck,
	"PlanckReduced":      PlanckReduced,
	"Boltzmann":          Boltzmann,
	"Avogadro":           Avogadro,
	"ElementaryCharge":   ElementaryCharge,
	"GravitationalConst": GravitationalConst,
	"VacuumPermittivity": VacuumPermittivity,
	"VacuumPermeability": VacuumPermeability,
	"StefanBoltzmann":    StefanBoltzmann,
	"GasConstant":        GasConstant,
	"StandardGravity":    StandardGravity,
	"AtmPressure":        AtmPressure,
}

// TestGoldenPhysicsConstants validates all physics constants against the
// golden-file test vectors in testdata/constants/physics_constants.json.
// This is the canonical cross-language validation — the same JSON file
// will be used by Python, C++, and C# implementations.
func TestGoldenPhysicsConstants(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/constants/physics_constants.json")

	if gf.Function != "Constants.Physics" {
		t.Errorf("golden file function = %q, want %q", gf.Function, "Constants.Physics")
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			name, ok := tc.Inputs["constant"].(string)
			if !ok {
				t.Fatalf("input 'constant' is not a string: %v", tc.Inputs["constant"])
			}

			got, exists := physicsConstantMap[name]
			if !exists {
				t.Fatalf("unknown constant %q in golden file", name)
			}

			testutil.AssertFloat64(t, tc, got)
		})
	}
}
