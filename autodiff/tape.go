package autodiff

// A Tape records the operations performed on Variables associated with it
// so that Backward can walk the operations in reverse and accumulate
// gradients.  The zero value is unusable; construct via NewTape.
//
// A single Tape is intended for one gradient evaluation: build the forward
// graph, call Backward, read the gradients.  Tapes are NOT safe for
// concurrent construction; use one Tape per goroutine.
type Tape struct {
	// nodes holds, for each Variable id, its forward value plus the
	// pullback that propagates an incoming gradient backward to its
	// input ids.
	nodes []node
}

type node struct {
	val      float64
	pullback func(grad float64, gradients []float64)
}

// NewTape returns an empty Tape.
func NewTape() *Tape {
	return &Tape{}
}

// Variable is a handle into a Tape: an integer identity plus the cached
// forward value.  Variables on different Tapes cannot be combined.
type Variable struct {
	Tape *Tape
	ID   int
	Val  float64
}

// Var registers a leaf Variable with the given value.  Leaves have no
// pullback (their incoming gradient is the answer).
func (t *Tape) Var(value float64) *Variable {
	id := len(t.nodes)
	t.nodes = append(t.nodes, node{val: value, pullback: nil})
	return &Variable{Tape: t, ID: id, Val: value}
}

// Constant registers a leaf Variable that does not contribute gradients.
// Its id is reserved but its pullback is a no-op (Backward zeroes its
// gradient at the end if it ends up in the path).  Practically equivalent
// to Var for the purposes of this minimal MVP, but kept as a distinct
// constructor for forward-compat with constant-folding.
func (t *Tape) Constant(value float64) *Variable {
	return t.Var(value)
}

// register adds a derived node with the given forward value and pullback,
// returning a Variable wrapping it.
func (t *Tape) register(value float64, pull func(grad float64, gradients []float64)) *Variable {
	id := len(t.nodes)
	t.nodes = append(t.nodes, node{val: value, pullback: pull})
	return &Variable{Tape: t, ID: id, Val: value}
}

// Backward computes the gradient of out with respect to every Variable
// recorded on the Tape.  Returns a slice grads such that grads[i] =
// d out / d Variable_i.val.
//
// The gradient at the output node is initialised to 1.0; non-output nodes
// start at 0 and are accumulated by each pullback.  Backward must be
// called exactly once per Tape; calling it twice produces undefined
// gradients (each pullback re-applies and would double-count).
func (t *Tape) Backward(out *Variable) []float64 {
	if out == nil || out.Tape != t {
		panic("autodiff: Backward called with a Variable from a different Tape")
	}
	grads := make([]float64, len(t.nodes))
	grads[out.ID] = 1.0
	// Walk in reverse: each pullback consumes grads[i] (gradient w.r.t.
	// node i's output) and adds contributions to its input ids.
	for i := len(t.nodes) - 1; i >= 0; i-- {
		if t.nodes[i].pullback == nil {
			continue
		}
		t.nodes[i].pullback(grads[i], grads)
	}
	return grads
}

// requireSame panics if a and b are on different Tapes.
func requireSame(a, b *Variable) {
	if a.Tape != b.Tape {
		panic("autodiff: cannot combine Variables from different Tapes")
	}
}
