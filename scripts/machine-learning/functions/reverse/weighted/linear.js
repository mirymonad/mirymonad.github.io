import { WeightedReverseFunction } from "../reverse.js";
import { NumTensor } from "../../../tensor.js";

/**
 * Linear layer weighted reverse function.
 */
export class Linear extends WeightedReverseFunction {

    constructor(inFeatures, outFeatures) {
        super(inFeatures, outFeatures);
        
        this.he();
    }

    /**
     * Computes the linear layer forward pass.
     * 
     * Definition: `f(x) = Wx`
     * 
     * @param {NumTensor} tensor the input vector (x).
     * @returns the unactivated output vector (z).
     */
    forwards(tensor) {
        return super.forwards(this.weights.t_mul(tensor));
    }
    
    /**
     * Computes the gradient of the loss with respect to each input (a_i).
     * 
     * By the chain rule:
     * 
     * `dL/da_i = Σ_k (dL/dz_k) * (dz_k/da_i)`
     * 
     * For example, given a linear layer with 2 inputs and 3 outputs:
     * 
     * ```
     * x1w1 + x2w2 = z1
     * x1w3 + x2w4 = z2
     * x1w5 + x2w6 = z3
     * ```
     * 
     * The gradient is the sum of the mapped weights multiplied by (dL / dz_k):
     * 
     * ```
     * dL/da1 = (dL/dz1) * w1 + (dL/dz2) * w3 + (dL/dz3) * w5
     * dL/da2 = (dL/dz1) * w2 + (dL/dz2) * w4 + (dL/dz3) * w6
     * ```
     * 
     * This is the transposed weight matrix multiplied to the (dL / dz) vector:
     * 
     * `(dL / da) = Wᵀ * (dL / dz)`
     * 
     * @returns A vector of the same size as the inputs containing their gradients.
     */
    backwards() {
        return this.weights.permute(1, 0).t_mul(super.backwards());
    }

    /**
     * Computes the gradient of the loss with respect to each weight (w_i).
     * 
     * By the chain rule:
     * 
     * `dL/dw_i = (dL/dz_i) * (dz_i/dw_i)`
     * 
     * For example, given a linear layer with 2 inputs and 3 outputs:
     * 
     * ```
     * x1w1 + x2w2 = z1
     * x1w3 + x2w4 = z2
     * x1w5 + x2w6 = z3
     * ```
     * 
     * The gradient is the mapped input multiplied by (dL / dz_i):
     * 
     * ```
     * dL / dw1 = dL / dz1 * x1
     * dL / dw2 = dL / dz1 * x2
     * dL / dw3 = dL / dz2 * x1
     * dL / dw4 = dL / dz2 * x2
     * dL / dw5 = dL / dz3 * x1
     * dL / dw6 = dL / dz3 * x2 
     * ```
     * 
     * This is the outer product of (dL / dz) vector applied to the input vector:
     * 
     * `(dL / dW) = (dL / dz).outer(x)`
     * 
     * @returns A tensor with the same shape as the weights containing their gradients.
     */
    cacheBackwards(gradient) {
        return gradient.outer(this.forwardsCache);
    }
}
