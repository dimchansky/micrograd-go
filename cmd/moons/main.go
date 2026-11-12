// Command moons trains a small MLP to separate two interleaving half-circles
// ("two moons"), a data set that no straight line can split, and draws the
// learned decision regions together with the loss curve.
//
// Regenerate the picture used in the README, next to this file:
//
//	go generate ./cmd/moons
package main

//go:generate go run . -out moons.png

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"path/filepath"

	mg "github.com/dimchansky/micrograd-go"
)

type options struct {
	samples, steps, grid, batch int
	noise, l2                   float64
	seed                        uint64
	out                         string
}

func (o options) validate() error {
	switch {
	case o.samples < 4:
		return fmt.Errorf("samples must be at least 4")
	case o.batch < 0:
		return fmt.Errorf("batch must not be negative")
	case o.steps < 0:
		return fmt.Errorf("steps must not be negative")
	case o.grid < 2:
		return fmt.Errorf("grid must be at least 2")
	case o.noise < 0 || math.IsNaN(o.noise) || math.IsInf(o.noise, 0):
		return fmt.Errorf("noise must be finite and nonnegative")
	case o.l2 < 0 || math.IsNaN(o.l2) || math.IsInf(o.l2, 0):
		return fmt.Errorf("l2 must be finite and nonnegative")
	case o.out == "":
		return fmt.Errorf("out must not be empty")
	}
	return nil
}

func main() {
	var o options
	flag.IntVar(&o.samples, "samples", 100, "number of training points")
	flag.IntVar(&o.batch, "batch", 0, "microbatch size (0 = all samples); one update per dataset pass")
	flag.IntVar(&o.steps, "steps", 100, "SGD steps (0 draws an untrained model)")
	flag.IntVar(&o.grid, "grid", 120, "decision-field cells across the plot")
	flag.Float64Var(&o.noise, "noise", 0.1, "standard deviation of Gaussian coordinate noise")
	flag.Float64Var(&o.l2, "l2", 1e-4, "L2 coefficient (all weights and biases)")
	flag.Uint64Var(&o.seed, "seed", 1337, "reproducible Go random seed")
	flag.StringVar(&o.out, "out", "moons.png", "output PNG path (replaced if it exists)")
	flag.Parse()
	log.SetFlags(0)
	if err := run(o); err != nil {
		log.Fatal(err)
	}
}

func run(o options) error {
	if err := o.validate(); err != nil {
		return err
	}
	data := makeMoons(o.samples, o.noise, rand.New(rand.NewPCG(o.seed, 0)))
	model := mg.NewMLP(2, []int{16, 16, 1}, rand.New(rand.NewPCG(o.seed, 1)))
	fmt.Printf("Two moons: samples=%d noise=%.3f seed=%d\n", len(data), o.noise, o.seed)
	fmt.Printf("MLP: 2 -> 16 -> 16 -> 1, parameters=%d\n", len(model.Parameters()))
	history, err := train(model, data, o.steps, o.batch, o.l2, func(step int, m metrics) {
		if step%10 == 0 || step == o.steps {
			fmt.Printf("step %3d: loss=%.6f (hinge=%.6f, L2=%.6f), training accuracy=%.1f%%\n",
				step, m.total(), m.dataLoss, m.regLoss, m.accuracy*100)
		}
	})
	if err != nil {
		return err
	}
	fmt.Println("Rendering decision regions and loss curve...")
	if err := render(o, model, data, history); err != nil {
		return err
	}
	path, err := filepath.Abs(o.out)
	if err != nil {
		return err
	}
	fmt.Println("Saved:", path)
	return nil
}

type sample struct {
	x, y, label float64
}

// makeMoons returns n points on two interleaving half-circles: the upper one
// centred at (0, 0) with label -1, and the lower one centred at (1, 0.5) with
// label +1. Gaussian noise blurs the points and the result is shuffled. This is
// the same geometry as scikit-learn's make_moons, but the random numbers come
// from Go, so the points differ from Python's for the same seed.
func makeMoons(n int, noise float64, rng *rand.Rand) []sample {
	data := make([]sample, 0, n)
	for class, count := range []int{n / 2, n - n/2} {
		for i := 0; i < count; i++ {
			angle := math.Pi * float64(i) / float64(max(count-1, 1))
			x, y, label := math.Cos(angle), math.Sin(angle), -1.0
			if class == 1 {
				x, y, label = 1-x, 0.5-y, 1
			}
			data = append(data, sample{
				x: x + noise*rng.NormFloat64(), y: y + noise*rng.NormFloat64(), label: label,
			})
		}
	}
	rng.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	return data
}

type metrics struct {
	dataLoss, regLoss, accuracy float64
}

func (m metrics) total() float64 { return m.dataLoss + m.regLoss }

func forward(model *mg.MLP, x, y float64) *mg.Value {
	return model.Forward([]*mg.Value{mg.New(x), mg.New(y)})[0]
}

// hinge is the loss max(0, 1 - label·score) for labels -1 and +1.
//
// label·score is positive when the sign of the score is right. The loss is
// zero once that margin reaches 1, so confidently correct points stop pushing
// the weights, and it grows linearly the more wrong the score is. Its gradient
// with respect to the score is -label inside the margin and 0 outside of it.
func hinge(score *mg.Value, label float64) *mg.Value {
	return mg.New(1).Add(score.Mul(mg.New(-label))).ReLU()
}

// objective returns mean hinge loss + alpha·Σ p² over all parameters p.
//
// The second term is L2 regularization. Its gradient 2·alpha·p pulls every
// parameter slightly towards zero, which keeps the weights small and the
// decision boundary smooth. Biases are included for simplicity. The score
// is only an unbounded margin; it is not a probability.
func objective(model *mg.MLP, data []sample, alpha float64) (*mg.Value, metrics) {
	loss := mg.New(0)
	correct := 0
	for _, p := range data {
		score := forward(model, p.x, p.y)
		loss = loss.Add(hinge(score, p.label))
		if (score.Data > 0) == (p.label > 0) {
			correct++
		}
	}
	loss = loss.Mul(mg.New(1 / float64(len(data))))
	reg := mg.New(0)
	for _, p := range model.Parameters() {
		reg = reg.Add(p.Pow(2))
	}
	reg = reg.Mul(mg.New(alpha))
	m := metrics{loss.Data, reg.Data, float64(correct) / float64(len(data))}
	return loss.Add(reg), m
}

// train runs steps updates of gradient descent and returns steps+1 history
// entries: entry k describes the model after k updates.
//
// Every step makes exactly one update from the gradient of the whole data set.
// With batchSize > 0 that gradient is computed in pieces: each microbatch gets
// its own fresh graph, its mean loss is scaled by its share of the data
// (n_batch/N), and Backward adds the result into the shared weight leaves.
// The sum of the scaled means equals the full-data mean, so the update is the
// same as with one big graph, but no single graph has to hold all the data.
//
// The learning rate decays linearly from 1.0 to 0.1, large steps first to
// make progress, small steps later to settle.
func train(model *mg.MLP, data []sample, steps, batchSize int, alpha float64, report func(int, metrics)) ([]metrics, error) {
	if len(data) == 0 || steps < 0 || batchSize < 0 {
		return nil, fmt.Errorf("training requires data and nonnegative steps/batch size")
	}
	if batchSize == 0 || batchSize > len(data) {
		batchSize = len(data)
	}
	params := model.Parameters()
	history := make([]metrics, 0, steps+1)
	for step := 0; step <= steps; step++ {
		// Zero once per update, not once per microbatch.
		for _, p := range params {
			p.Grad = 0
		}
		var total metrics
		for start := 0; start < len(data); start += batchSize {
			end := min(start+batchSize, len(data))
			loss, m := objective(model, data[start:end], alpha)
			if math.IsNaN(m.total()) || math.IsInf(m.total(), 0) {
				return nil, fmt.Errorf("non-finite loss at step %d", step)
			}
			weight := float64(end-start) / float64(len(data))
			total.dataLoss += weight * m.dataLoss
			total.regLoss += weight * m.regLoss
			total.accuracy += weight * m.accuracy
			if step < steps {
				// Fresh graph; parameter gradients accumulate through +=.
				loss.Mul(mg.New(weight)).Backward()
			}
		}
		history = append(history, total)
		if report != nil {
			report(step, total)
		}
		if step < steps {
			rate := 1 - 0.9*float64(step)/float64(steps)
			for _, p := range params {
				p.Data -= rate * p.Grad
			}
		}
	}
	return history, nil
}
