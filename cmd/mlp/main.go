// Command mlp learns XOR with a 2 -> 8 -> 1 neural network.
package main

import (
	"fmt"
	"math/rand/v2"

	mg "github.com/dimchansky/micrograd-go"
)

func main() {
	model := mg.NewMLP(2, []int{8, 1}, rand.New(rand.NewPCG(1, 2)))
	params := model.Parameters()
	xs := [][2]float64{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}}
	ys := []float64{-1, 1, 1, -1}
	fmt.Printf("XOR: 2 -> 8 -> 1, parameters: %d\n", len(params))

	for epoch := 0; epoch < 500; epoch++ {
		loss := mg.New(0)
		for i, x := range xs {
			out := model.Forward([]*mg.Value{mg.New(x[0]), mg.New(x[1])})
			loss = loss.Add(out[0].Sub(mg.New(ys[i])).Pow(2))
		}
		loss = loss.Div(mg.New(float64(len(xs))))

		for _, p := range params {
			p.Grad = 0
		}
		loss.Backward()
		for _, p := range params {
			p.Data -= 0.05 * p.Grad
		}

		if epoch%100 == 0 || epoch == 499 {
			fmt.Printf("epoch %3d: MSE = %.8f\n", epoch, loss.Data)
		}
	}
	for i, x := range xs {
		out := model.Forward([]*mg.Value{mg.New(x[0]), mg.New(x[1])})
		fmt.Printf("%v => %+.6f (target %+.0f)\n", x, out[0].Data, ys[i])
	}
}
