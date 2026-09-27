import "https://cdn.plot.ly/plotly-4.1.1.min.js"

import { Linear } from "./functions/reverse/weighted/linear.js";
import { ReLU } from "./functions/reverse/activation/relu.js";
import { MSE } from "./functions/loss/mse.js";
import { SGD } from "./optimisers/sgd.js";
import { NumTensor } from "./tensor.js";
import { Network } from "./network.js";

function genGuassianBell()  {
    const x = [];
    const y = [];
    const z = [];

    // Initialise the coords for the input
    for (let i = -1; i <= 1; i += 0.04) {
        x.push(i);
        y.push(i);
    }

    // Calculate the coords for the output
    for (let i = 0; i < y.length; i++) {
        const row = [];

        for (let j = 0; j < x.length; j++) {
            row.push(0.1 * Math.exp(-(x[j] * x[j] + y[i] * y[i])));
        }

        z.push(row);
    }

    return {
        x: x,
        y: y,
        z: z,
        name: "Target",
        type: 'surface',
        opacity: 1,
        visible: true
    }
}

function genCurrent(func) {
    const x = [];
    const y = [];
    const z = [];

    // Initialise the coords for the input
    for (let i = -1; i < 1; i += 0.04) {
        x.push(i);
        y.push(i);
    }

    // Calculate the coords for the output
    for (let i = 0; i < y.length; i++) {
        const row = [];

        for (let j = 0; j < x.length; j++) {
            let X = new NumTensor([2], new Float32Array([x[j], y[i]]));

            // Predict the z-coordinate using the trained neural network
            row.push(func.forwards(X)._data[0]);
        }

        z.push(row);
    }

    return {
        x: x,
        y: y,
        z: z,
        name: "Current",
        type: 'surface',
        opacity: 0.15,
        visible: true
    };
}

/**
 * ─────────────────────────────────────────────
 * Custom Neural Network
 * ─────────────────────────────────────────────
 */
const loss = new MSE();

const func = Network.sequential(
  loss,
  new Linear(2, 1),
  new ReLU(),
  new Linear(2, 2),
);

const optimiser = new SGD(0.1);

/**
 * ─────────────────────────────────────────────
 * Plotly 3D Plot
 * ─────────────────────────────────────────────
 */
const custom = document.getElementById("custom-func");

const scatter = {
    x: [],
    y: [],
    z: [],
    type: "scatter3d",
    mode: "markers",
    marker: {
        size: 5,
        color: "green"
    }
};

const layout = {
    title: {
        text: "Click the Target to add training data to the Current neural network"
    },
};

Plotly.newPlot(custom, [genGuassianBell(), genCurrent(func), scatter], layout);


/**
 * ─────────────────────────────────────────────
 * User Interaction
 * ─────────────────────────────────────────────
 */
let update = false;

custom.on('plotly_click', function(event) {
    if (update) return;

    const point = event.points[0];

    // Ignore clicks on Surface 2
    if (point.curveNumber !== 0) {
        return;
    }

    update = true;
    
    const X = new NumTensor([2], new Float32Array([point.x, point.y]));
    const Y = new NumTensor([1], new Float32Array([point.z]));

    const predicted = func.forwards(X);
    loss.apply(predicted, Y);
    func.backwards();
    func.step(optimiser);

    Plotly.extendTraces(custom, {
        x: [[point.x]],
        y: [[point.y]],
        z: [[point.z]]
    }, [2]);

    const surface = genCurrent(func);

    Plotly.restyle(custom, {
        x: [surface.x],
        y: [surface.y],
        z: [surface.z]
    }, [1]);

    update = false;
});
