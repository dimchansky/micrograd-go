// Command demo shows one derivative, then learns y = 2*x + 1.
package main

import (
	"fmt"

	mg "example.com/micrograd"
)

func main() {
	x := mg.New(3)
	y := x.Mul(x).Add(x)
	y.Backward()
	fmt.Printf("y = %.0f, dy/dx = %.0f\n", y.Data, x.Grad)

	w, b := mg.New(0), mg.New(0)
	xs := []float64{-2, -1, 0, 1, 2}
	for step := 0; step < 100; step++ {
		// Forward: a fresh graph, reusing only the weight leaves.
		loss := mg.New(0)
		for _, x := range xs {
			prediction := w.Mul(mg.New(x)).Add(b)
			loss = loss.Add(prediction.Sub(mg.New(2*x + 1)).Pow(2))
		}
		loss = loss.Div(mg.New(float64(len(xs))))

		// Backward: clear old gradients, then apply the chain rule.
		w.Grad, b.Grad = 0, 0
		loss.Backward()

		// Update: ordinary arithmetic, outside the computation graph.
		w.Data -= 0.1 * w.Grad
		b.Data -= 0.1 * b.Grad
		if step == 0 || step == 99 {
			fmt.Printf("step %d: loss = %.10f\n", step, loss.Data)
		}
	}
	fmt.Printf("learned: w = %.6f, b = %.6f\n", w.Data, b.Data)
}
