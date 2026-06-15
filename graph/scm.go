package graph

import (
	"math/rand"
	"sort"
)

// Discrete (binary) structural causal models over an ADMG, with exact
// enumeration of observational and interventional distributions, plus a
// numerical evaluator for the symbolic functionals returned by the ID algorithm.
//
// This is both a capability (compute P(Y|do X) from a fully-specified model) and
// the capstone validation of idalgorithm.go: a model's TRUE interventional
// distribution (via the truncated factorization) and the ID algorithm's SYMBOLIC
// functional (evaluated against the model's observational joint) are two
// genuinely different computations; that they agree numerically validates the
// returned expression, not just the identifiability verdict.
//
// Variables are binary {0,1}. Each bidirected edge U<->V is realised as a shared
// latent binary parent of U and V (the canonical SCM of an ADMG).

// DiscreteSCM is a binary SCM consistent with an ADMG. Build with RandomSCM.
type DiscreteSCM struct {
	observed []string            // sorted observed vertices
	latents  []string            // one binary latent per bidirected edge
	parents  map[string][]string // node -> ordered parents (observed dir. parents + latent parents)
	p1       map[string]map[string]float64 // node -> parentConfigKey -> P(node=1 | parents)
	platent  map[string]float64  // latent -> P(=1)
}

// RandomSCM constructs a random binary SCM consistent with g: each bidirected
// edge becomes a fresh latent Bernoulli(0.5) parent of both endpoints, and every
// observed node gets a random Bernoulli CPT entry per parent configuration. The
// seed makes it deterministic.
func RandomSCM(g ADMG, seed int64) DiscreteSCM {
	rng := rand.New(rand.NewSource(seed))
	scm := DiscreteSCM{
		observed: append([]string{}, g.nodes...),
		parents:  map[string][]string{},
		p1:       map[string]map[string]float64{},
		platent:  map[string]float64{},
	}
	sort.Strings(scm.observed)
	// directed parents
	dirParents := map[string][]string{}
	for _, e := range g.directed {
		dirParents[e[1]] = append(dirParents[e[1]], e[0])
	}
	// a latent per bidirected edge, parent of both endpoints
	latParents := map[string][]string{}
	for i, e := range g.bidirected {
		l := "__L" + itoa(i)
		scm.latents = append(scm.latents, l)
		scm.platent[l] = 0.5
		latParents[e[0]] = append(latParents[e[0]], l)
		latParents[e[1]] = append(latParents[e[1]], l)
	}
	for _, v := range scm.observed {
		ps := append(append([]string{}, dirParents[v]...), latParents[v]...)
		sort.Strings(ps)
		scm.parents[v] = ps
		scm.p1[v] = map[string]float64{}
		enumerate(ps, func(cfg map[string]int) {
			// random CPT entry strictly inside (0,1) to avoid degenerate division
			scm.p1[v][cfgKey(cfg, ps)] = 0.05 + 0.9*rng.Float64()
		})
	}
	return scm
}

// mechanism returns P(node = val | parent assignment).
func (s DiscreteSCM) mechanism(node string, val int, assign map[string]int) float64 {
	p := s.p1[node][cfgKey(assign, s.parents[node])]
	if val == 1 {
		return p
	}
	return 1 - p
}

// ObservationalJoint returns P(observed = a) for every full observed assignment,
// keyed canonically. Latents are summed out.
func (s DiscreteSCM) ObservationalJoint() map[string]float64 {
	joint := map[string]float64{}
	enumerate(s.observed, func(obs map[string]int) {
		key := cfgKey(obs, s.observed)
		sum := 0.0
		enumerate(s.latents, func(lat map[string]int) {
			full := merge(obs, lat)
			w := 1.0
			for _, v := range s.observed {
				w *= s.mechanism(v, full[v], full)
			}
			for _, l := range s.latents {
				if lat[l] == 1 {
					w *= s.platent[l]
				} else {
					w *= 1 - s.platent[l]
				}
			}
			sum += w
		})
		joint[key] = sum
	})
	return joint
}

// InterventionalDistribution returns the true P(outcome = . | do(treatment=xval))
// by the truncated factorization (delete the mechanisms of intervened nodes, fix
// them to xval), keyed by the outcome assignment.
func (s DiscreteSCM) InterventionalDistribution(outcome []string, xval map[string]int) map[string]float64 {
	out := map[string]float64{}
	nonX := []string{}
	for _, v := range s.observed {
		if _, fixed := xval[v]; !fixed {
			nonX = append(nonX, v)
		}
	}
	ySorted := append([]string{}, outcome...)
	sort.Strings(ySorted)
	enumerate(nonX, func(rest map[string]int) {
		obs := merge(rest, xval)
		sum := 0.0
		enumerate(s.latents, func(lat map[string]int) {
			full := merge(obs, lat)
			w := 1.0
			for _, v := range nonX { // intervened nodes contribute no mechanism factor
				w *= s.mechanism(v, full[v], full)
			}
			for _, l := range s.latents {
				if lat[l] == 1 {
					w *= s.platent[l]
				} else {
					w *= 1 - s.platent[l]
				}
			}
			sum += w
		})
		out[cfgKey(obs, ySorted)] += sum
	})
	return out
}

// EvalFunctional evaluates a symbolic ID expression e at a full assignment of its
// free variables, against the observational joint. jointMarginal sums the joint
// over unspecified observed variables.
func evalFunctional(e expr, assign map[string]int, joint map[string]float64, observed []string) float64 {
	switch t := e.(type) {
	case *exprFactor:
		num := jointMarginal(joint, observed, sub(assign, append(append([]string{}, t.vars...), t.given...)))
		if len(t.given) == 0 {
			return num
		}
		den := jointMarginal(joint, observed, sub(assign, t.given))
		if den == 0 {
			return 0
		}
		return num / den
	case *exprProduct:
		p := 1.0
		for _, f := range t.factors {
			p *= evalFunctional(f, assign, joint, observed)
		}
		return p
	case *exprMarginal:
		sum := 0.0
		enumerate(t.over, func(ext map[string]int) {
			sum += evalFunctional(t.inner, merge(assign, ext), joint, observed)
		})
		return sum
	case *exprP, exprP:
		return jointMarginal(joint, observed, assign)
	}
	return 0
}

// ---- enumeration / key helpers ----------------------------------------------

func enumerate(vars []string, fn func(map[string]int)) {
	n := len(vars)
	for mask := 0; mask < (1 << n); mask++ {
		a := make(map[string]int, n)
		for i, v := range vars {
			a[v] = (mask >> i) & 1
		}
		fn(a)
	}
}

func cfgKey(assign map[string]int, order []string) string {
	key := ""
	for _, v := range order {
		key += v + "=" + itoa(assign[v]) + ";"
	}
	return key
}

// jointMarginal sums the full observational joint over observed variables not
// fixed by `partial`.
func jointMarginal(joint map[string]float64, observed []string, partial map[string]int) float64 {
	free := []string{}
	for _, v := range observed {
		if _, ok := partial[v]; !ok {
			free = append(free, v)
		}
	}
	sum := 0.0
	enumerate(free, func(ext map[string]int) {
		sum += joint[cfgKey(merge(partial, ext), observed)]
	})
	return sum
}

func sub(assign map[string]int, vars []string) map[string]int {
	out := make(map[string]int, len(vars))
	for _, v := range vars {
		out[v] = assign[v]
	}
	return out
}

func merge(a, b map[string]int) map[string]int {
	out := make(map[string]int, len(a)+len(b))
	for k, v := range a {
		out[k] = v
	}
	for k, v := range b {
		out[k] = v
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}
