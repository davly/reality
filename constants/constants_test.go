package constants

import (
	"math"
	"testing"
)

// TestMathConstants verifies mathematical constants against known values.
func TestMathConstants(t *testing.T) {
	tests := []struct {
		name     string
		got      float64
		expected float64
	}{
		// 1. Pi matches Go stdlib
		{"Pi matches math.Pi", Pi, math.Pi},
		// 2. E matches Go stdlib
		{"E matches math.E", E, math.E},
		// 3. Phi is the golden ratio
		{"Phi = (1+sqrt(5))/2", Phi, (1 + math.Sqrt(5)) / 2},
		// 4. Sqrt2 matches Go stdlib
		{"Sqrt2 matches math.Sqrt2", Sqrt2, math.Sqrt2},
		// 5. Sqrt3 matches computation
		{"Sqrt3 = sqrt(3)", Sqrt3, math.Sqrt(3)},
		// 6. Ln2 matches Go stdlib
		{"Ln2 matches math.Ln2", Ln2, math.Ln2},
		// 7. Ln10 matches Go stdlib
		{"Ln10 matches math.Ln10", Ln10, math.Ln10},
		// 8. Log2E matches Go stdlib
		{"Log2E matches math.Log2E", Log2E, math.Log2E},
		// 9. Log10E matches Go stdlib
		{"Log10E matches math.Log10E", Log10E, math.Log10E},
		// 10. EulerGamma matches known value
		{"EulerGamma ~ 0.5772156649015329", EulerGamma, 0.5772156649015329},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.expected {
				t.Errorf("got %.20e, want %.20e (diff = %e)",
					tt.got, tt.expected, math.Abs(tt.got-tt.expected))
			}
		})
	}
}

// TestMathRelationships verifies known mathematical identities.
func TestMathRelationships(t *testing.T) {
	// 11. Log2E * Ln2 = 1
	t.Run("Log2E * Ln2 = 1", func(t *testing.T) {
		product := Log2E * Ln2
		if math.Abs(product-1.0) > 1e-15 {
			t.Errorf("Log2E * Ln2 = %.20e, want 1.0", product)
		}
	})

	// 12. Log10E * Ln10 = 1
	t.Run("Log10E * Ln10 = 1", func(t *testing.T) {
		product := Log10E * Ln10
		if math.Abs(product-1.0) > 1e-15 {
			t.Errorf("Log10E * Ln10 = %.20e, want 1.0", product)
		}
	})

	// 13. Phi^2 = Phi + 1 (golden ratio property)
	t.Run("Phi^2 = Phi + 1", func(t *testing.T) {
		phiSquared := Phi * Phi
		phiPlusOne := Phi + 1.0
		if math.Abs(phiSquared-phiPlusOne) > 1e-14 {
			t.Errorf("Phi^2 = %.20e, Phi+1 = %.20e", phiSquared, phiPlusOne)
		}
	})

	// 14. Sqrt2^2 = 2
	t.Run("Sqrt2^2 = 2", func(t *testing.T) {
		sq := Sqrt2 * Sqrt2
		if math.Abs(sq-2.0) > 1e-15 {
			t.Errorf("Sqrt2^2 = %.20e, want 2.0", sq)
		}
	})

	// 15. Sqrt3^2 = 3
	t.Run("Sqrt3^2 = 3", func(t *testing.T) {
		sq := Sqrt3 * Sqrt3
		if math.Abs(sq-3.0) > 1e-15 {
			t.Errorf("Sqrt3^2 = %.20e, want 3.0", sq)
		}
	})
}

// TestPhysicsConstants verifies physics constants against known values.
func TestPhysicsConstants(t *testing.T) {
	// 16. Speed of light is exactly 299792458 m/s
	t.Run("SpeedOfLight = 299792458", func(t *testing.T) {
		if SpeedOfLight != 299792458.0 {
			t.Errorf("SpeedOfLight = %v, want 299792458", SpeedOfLight)
		}
	})

	// 17. Planck constant SI 2019 exact
	t.Run("Planck = 6.62607015e-34", func(t *testing.T) {
		if Planck != 6.62607015e-34 {
			t.Errorf("Planck = %e, want 6.62607015e-34", Planck)
		}
	})

	// 18. Boltzmann SI 2019 exact
	t.Run("Boltzmann = 1.380649e-23", func(t *testing.T) {
		if Boltzmann != 1.380649e-23 {
			t.Errorf("Boltzmann = %e, want 1.380649e-23", Boltzmann)
		}
	})

	// 19. Avogadro SI 2019 exact
	t.Run("Avogadro = 6.02214076e23", func(t *testing.T) {
		if Avogadro != 6.02214076e23 {
			t.Errorf("Avogadro = %e, want 6.02214076e23", Avogadro)
		}
	})

	// 20. Elementary charge SI 2019 exact
	t.Run("ElementaryCharge = 1.602176634e-19", func(t *testing.T) {
		if ElementaryCharge != 1.602176634e-19 {
			t.Errorf("ElementaryCharge = %e, want 1.602176634e-19", ElementaryCharge)
		}
	})

	// 21. PlanckReduced = h / (2*pi)
	t.Run("PlanckReduced = h/(2*pi)", func(t *testing.T) {
		expected := Planck / (2 * math.Pi)
		if math.Abs(PlanckReduced-expected) > 1e-50 {
			t.Errorf("PlanckReduced = %e, want %e", PlanckReduced, expected)
		}
	})

	// 22. Standard gravity exact
	t.Run("StandardGravity = 9.80665", func(t *testing.T) {
		if StandardGravity != 9.80665 {
			t.Errorf("StandardGravity = %v, want 9.80665", StandardGravity)
		}
	})

	// 23. Atmospheric pressure exact
	t.Run("AtmPressure = 101325", func(t *testing.T) {
		if AtmPressure != 101325.0 {
			t.Errorf("AtmPressure = %v, want 101325", AtmPressure)
		}
	})

	// 24. GasConstant = Avogadro * Boltzmann
	t.Run("GasConstant = N_A * k_B", func(t *testing.T) {
		expected := 6.02214076e23 * 1.380649e-23
		if math.Abs(GasConstant-expected) > 1e-15 {
			t.Errorf("GasConstant = %.15e, want %.15e", GasConstant, expected)
		}
	})

	// 25. StefanBoltzmann known value
	t.Run("StefanBoltzmann ~ 5.670374419e-8", func(t *testing.T) {
		if math.Abs(StefanBoltzmann-5.670374419e-8) > 1e-17 {
			t.Errorf("StefanBoltzmann = %e, want 5.670374419e-8", StefanBoltzmann)
		}
	})

	// 26. GravitationalConst known value
	t.Run("GravitationalConst ~ 6.67430e-11", func(t *testing.T) {
		if math.Abs(GravitationalConst-6.67430e-11) > 1e-16 {
			t.Errorf("GravitationalConst = %e, want 6.67430e-11", GravitationalConst)
		}
	})

	// 27. VacuumPermittivity known value
	t.Run("VacuumPermittivity ~ 8.8541878128e-12", func(t *testing.T) {
		if math.Abs(VacuumPermittivity-8.8541878128e-12) > 1e-22 {
			t.Errorf("VacuumPermittivity = %e, want 8.8541878128e-12", VacuumPermittivity)
		}
	})

	// 28. VacuumPermeability known value
	t.Run("VacuumPermeability ~ 1.25663706212e-6", func(t *testing.T) {
		if math.Abs(VacuumPermeability-1.25663706212e-6) > 1e-17 {
			t.Errorf("VacuumPermeability = %e, want 1.25663706212e-6", VacuumPermeability)
		}
	})
}

// TestUnitConversions verifies unit conversion constants.
func TestUnitConversions(t *testing.T) {
	// 29. MetersPerMile = 1609.344
	t.Run("MetersPerMile = 1609.344", func(t *testing.T) {
		if MetersPerMile != 1609.344 {
			t.Errorf("MetersPerMile = %v, want 1609.344", MetersPerMile)
		}
	})

	// 30. MetersPerFoot = 0.3048
	t.Run("MetersPerFoot = 0.3048", func(t *testing.T) {
		if MetersPerFoot != 0.3048 {
			t.Errorf("MetersPerFoot = %v, want 0.3048", MetersPerFoot)
		}
	})

	// 31. MetersPerInch = 0.0254
	t.Run("MetersPerInch = 0.0254", func(t *testing.T) {
		if MetersPerInch != 0.0254 {
			t.Errorf("MetersPerInch = %v, want 0.0254", MetersPerInch)
		}
	})

	// 32. KgPerPound = 0.45359237
	t.Run("KgPerPound = 0.45359237", func(t *testing.T) {
		if KgPerPound != 0.45359237 {
			t.Errorf("KgPerPound = %v, want 0.45359237", KgPerPound)
		}
	})

	// 33. CelsiusToKelvin = 273.15
	t.Run("CelsiusToKelvin = 273.15", func(t *testing.T) {
		if CelsiusToKelvin != 273.15 {
			t.Errorf("CelsiusToKelvin = %v, want 273.15", CelsiusToKelvin)
		}
	})

	// 34. RadiansToDegrees * DegreesToRadians = 1
	t.Run("RadiansToDegrees * DegreesToRadians = 1", func(t *testing.T) {
		product := RadiansToDegrees * DegreesToRadians
		if math.Abs(product-1.0) > 1e-15 {
			t.Errorf("RadiansToDegrees * DegreesToRadians = %.20e, want 1.0", product)
		}
	})

	// 35. 90 degrees = Pi/2 radians
	t.Run("90 degrees = Pi/2 radians", func(t *testing.T) {
		radians := 90.0 * DegreesToRadians
		if math.Abs(radians-Pi/2) > 1e-15 {
			t.Errorf("90 * DegreesToRadians = %.20e, want Pi/2 = %.20e", radians, Pi/2)
		}
	})

	// 36. 1 mile = 5280 feet
	t.Run("1 mile = 5280 feet", func(t *testing.T) {
		mileInFeet := MetersPerMile / MetersPerFoot
		if math.Abs(mileInFeet-5280.0) > 1e-10 {
			t.Errorf("MetersPerMile / MetersPerFoot = %v, want 5280", mileInFeet)
		}
	})

	// 37. 1 foot = 12 inches
	t.Run("1 foot = 12 inches", func(t *testing.T) {
		footInInches := MetersPerFoot / MetersPerInch
		if math.Abs(footInInches-12.0) > 1e-10 {
			t.Errorf("MetersPerFoot / MetersPerInch = %v, want 12", footInInches)
		}
	})

	// 38. 1 ounce = 1/16 pound
	t.Run("1 ounce = 1/16 pound", func(t *testing.T) {
		ratio := KgPerOunce / KgPerPound
		if math.Abs(ratio-1.0/16.0) > 1e-15 {
			t.Errorf("KgPerOunce / KgPerPound = %v, want 1/16", ratio)
		}
	})

	// 39. AtmPressure = PascalsPerAtm
	t.Run("AtmPressure == PascalsPerAtm", func(t *testing.T) {
		if AtmPressure != PascalsPerAtm {
			t.Errorf("AtmPressure (%v) != PascalsPerAtm (%v)", AtmPressure, PascalsPerAtm)
		}
	})

	// 40. Seconds per day = 60 * 60 * 24
	t.Run("SecondsPerDay = 86400", func(t *testing.T) {
		if SecondsPerDay != 86400.0 {
			t.Errorf("SecondsPerDay = %v, want 86400", SecondsPerDay)
		}
	})
}
