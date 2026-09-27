import { NumTensor } from "../tensor.js";

export class SGD {

    /**
     * @param {number} lr the learning rate 
     */
    constructor(lr) {
        this.lr = lr;
    }

    /**
     * Updates the input weights based on their respective gradient.
     * 
     * Definition: `W -= lr * (dL / dW)`
     * 
     * @param {NumTensor} weights the input weights.
     * @param {NumTensor} gradients the weights gradients.
     * @returns The updated weight tensor of the same shape.
     */
    optimise(weights, gradients) {
        return weights.sub(gradients.mul(this.lr));
    }
}
