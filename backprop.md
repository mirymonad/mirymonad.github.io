---
permalink: /backprop
layout: default
title: Backpropagation Visualisation
---

<div id="custom-func"></div>

A visualisation of a 2D Guassian Bell shaped function f(x,y)=0.1e^−(x^2+y^2) attempted to be fit by a neural network which takes takes two inputs, has a single hidden layer with two ReLU activated perceptrons, and one output.

## Instructions

---

1. Click on any point on the 2D Guassian Bell shaped function (Target).

2. The point will leave a green marker behind, and be used to update the weights of the neural network function via backproapgation (Current).

3. Add points and see how close you can train neural network to match!

---

## End Remarks

Please note that due to the mathematical definition of the neural network, it is impossible to set the weights to exactly match the Guassian Bell equation.

<script type="module" src="scripts/machine-learning/backprop.js" defer></script>