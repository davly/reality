// Package idbench is a TEST/EVAL-ONLY instrument that measures how well the
// audio.Fingerprint primitive discriminates INDIVIDUAL entities (the Aurora
// "which one, not what kind" thesis) — birds, pets, machines. It is the
// discrimination harness, NOT a shipping path and NOT the real-data verdict:
// it computes intra- vs inter-individual distance separation, an equal-error
// rate, and rank-1 closed-set accuracy over labeled feature vectors, using an
// honest enroll/probe split (a probe is never scored against a fingerprint it
// helped build). The real-world verdict is data-gated (it needs a labelled
// multi-individual / multi-day holdout that does not exist on disk yet); this
// instrument runs green on synthetic data so the METRIC is verifiable before any
// corpus is acquired.
package idbench

import (
	"errors"
	"sort"

	"github.com/davly/reality/audio"
)

// Sample is one labeled feature vector (e.g. an MFCC frame) with the true
// individual it came from.
type Sample struct {
	ID string
	X  []float64
}

// Report holds the discrimination metrics for one evaluation.
type Report struct {
	NIndividuals    int
	NEnroll         int
	NProbe          int
	GenuineMean     float64 // mean distance: probe vs its OWN-individual fingerprint
	ImpostorMean    float64 // mean distance: probe vs its nearest OTHER-individual fingerprint
	IntraInterRatio float64 // GenuineMean / ImpostorMean (a good fingerprint is << 1)
	EER             float64 // equal error rate over the genuine/impostor distance distributions
	Threshold       float64 // distance threshold at the EER
	Rank1Accuracy   float64 // fraction of probes whose globally-nearest fingerprint is their own id
}

// Evaluate builds one fingerprint per individual from enroll, then scores each
// probe against every fingerprint. Returns the discrimination Report. epsilon is
// the variance-floor passed to audio.FingerprintMahalanobis. Pure function — no
// I/O, no model calls.
func Evaluate(enroll, probe []Sample, dim int, epsilon float64) (Report, error) {
	if dim < 1 {
		return Report{}, errors.New("idbench: dim must be >= 1")
	}
	if len(enroll) == 0 || len(probe) == 0 {
		return Report{}, errors.New("idbench: enroll and probe must be non-empty")
	}

	// Build per-individual fingerprints from the enroll split (stable id order).
	fps := map[string]*audio.Fingerprint{}
	var ids []string
	for _, s := range enroll {
		if len(s.X) != dim {
			return Report{}, errors.New("idbench: enroll sample dimension mismatch")
		}
		fp, ok := fps[s.ID]
		if !ok {
			f := audio.NewFingerprint(dim)
			fp = &f
			fps[s.ID] = fp
			ids = append(ids, s.ID)
		}
		audio.UpdateFingerprint(fp, s.X)
	}
	sort.Strings(ids)

	var genuine, impostorAll, nearestImpostor []float64
	rank1Correct, rank1Total := 0, 0

	for _, p := range probe {
		if len(p.X) != dim {
			return Report{}, errors.New("idbench: probe sample dimension mismatch")
		}
		ownFP, ownKnown := fps[p.ID]
		bestID, bestDist := "", 0.0
		first := true
		nearImp := 0.0
		haveImp := false

		for _, id := range ids {
			d := audio.FingerprintMahalanobis(fps[id], p.X, epsilon)
			if first || d < bestDist {
				bestDist, bestID, first = d, id, false
			}
			if id == p.ID {
				continue
			}
			impostorAll = append(impostorAll, d)
			if !haveImp || d < nearImp {
				nearImp, haveImp = d, true
			}
		}

		if ownKnown {
			genuine = append(genuine, audio.FingerprintMahalanobis(ownFP, p.X, epsilon))
			rank1Total++
			if bestID == p.ID {
				rank1Correct++
			}
		}
		if haveImp {
			nearestImpostor = append(nearestImpostor, nearImp)
		}
	}

	rep := Report{
		NIndividuals: len(fps),
		NEnroll:      len(enroll),
		NProbe:       len(probe),
	}
	rep.GenuineMean = mean(genuine)
	rep.ImpostorMean = mean(nearestImpostor)
	if rep.ImpostorMean > 0 {
		rep.IntraInterRatio = rep.GenuineMean / rep.ImpostorMean
	}
	if rank1Total > 0 {
		rep.Rank1Accuracy = float64(rank1Correct) / float64(rank1Total)
	}
	rep.EER, rep.Threshold = eer(genuine, impostorAll)
	return rep, nil
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// eer sweeps candidate distance thresholds and returns the equal-error rate and
// the threshold where the false-accept rate (impostor distance <= t) most nearly
// equals the false-reject rate (genuine distance > t). Distances: lower == more
// genuine. Returns (0.5, 0) if either distribution is empty.
func eer(genuine, impostor []float64) (float64, float64) {
	if len(genuine) == 0 || len(impostor) == 0 {
		return 0.5, 0
	}
	cands := make([]float64, 0, len(genuine)+len(impostor))
	cands = append(cands, genuine...)
	cands = append(cands, impostor...)
	sort.Float64s(cands)

	bestEER, bestT, bestGap := 0.5, cands[0], 2.0
	for _, t := range cands {
		far := frac(impostor, func(d float64) bool { return d <= t }) // impostor accepted
		frr := frac(genuine, func(d float64) bool { return d > t })   // genuine rejected
		gap := far - frr
		if gap < 0 {
			gap = -gap
		}
		if gap < bestGap {
			bestGap, bestEER, bestT = gap, (far+frr)/2, t
		}
	}
	return bestEER, bestT
}

func frac(xs []float64, pred func(float64) bool) float64 {
	if len(xs) == 0 {
		return 0
	}
	c := 0
	for _, x := range xs {
		if pred(x) {
			c++
		}
	}
	return float64(c) / float64(len(xs))
}
