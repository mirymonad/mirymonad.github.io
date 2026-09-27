import { ReverseFunction } from "../reverse.js";
import { NumTensor } from "../../../tensor.js";

/**
 * Rectified Linear Unit (ReLU) activation function.
 */
export class ReLU extends ReverseFunction {

    /**
     * Computes the ReLU forward pass.
     * 
     * Definition: `ReLU(z) = max(0, z)`
     * 
     * @param {NumTensor} tensor the input vector (z).
     * @returns the activated output vector (x).
     */
    forwards(tensor) {
        const result = new NumTensor(tensor.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = Math.max(0, tensor._data[i]);
        }

        return super.forwards(result);
    }

    /**
     * Computes the gradient of the loss with respect to each input `(z_i)`.
     * 
     * By the chain rule:
     * 
     * `dL/dz_i = (dL/da_i) * (da_i/dz_i)`
     * 
     * For example, given the input vector:
     * 
     * `z = (10, 0, 5)`
     * 
     * The gradient is `(dL/da_i)` if `z_i > 0`:
     * 
     * ```
     * dL / dz1 = (dL/da1)
     * dL / dz2 = 0
     * dL / dz3 = (dL/da3)
     * ```
     * 
     * This is the hadamard of the ReLU derivative and the `(dL / da)` vector:
     * 
     * `ReLU'(z).hadamard(dL / da)`
     * 
     * @returns A vector of the same size as the inputs containing their gradients.
     */
    backwards() {
        const result = new NumTensor(this.forwardsCache.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this.forwardsCache._data[i] > 0 ? 1 : 0;
        }

        return result.hadamard(super.backwards());
    }
}
