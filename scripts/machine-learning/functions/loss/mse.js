import { NumTensor } from "../../tensor.js";
import { ReverseLossFunction } from "./loss.js";

/**
 * Mean Squared Error (MSE) loss function. 
 */
export class MSE extends ReverseLossFunction {

    /**
     * Calculates the loss between the predicted and expected.
     * 
     * Definition: `MSE(ŷ, y) = (1/n) * Σ_i (ŷ_i - y_i)^2
     * 
     * @param {NumTensor} predicted 
     * @param {*} expected 
     * @returns 
     */
    apply(predicted, expected) {
        return predicted.sub(expected).pow(2).average;
    }

    /**
     * Computes the gradient of loss with respect to each predicted input `(ŷ_i)`.
     * 
     * Definition: `MSE'(ŷ, y) = (2/n) * (ŷ-y)`
     * 
     * For example, given the predicted input vector:
     * 
     * `ŷ = (10, 0, 5)`
     * 
     * The gradient is `(2/n) * (ŷ_i-y_i)` for each predicted input:
     * 
     * ```
     * ŷ1 = 2/3 * (10-y)
     * ŷ2 = 2/3 * (0-y)
     * ŷ3 = 2/3 * (5-y)
     * ```
     * 
     * This is the subtraction of the input vectors multiplied by `(2/n)`:
     * 
     * `(ŷ - y) * (2 / n)`
     * 
     * @returns A vector of the same size as the inputs containing their gradients.
     */
    backwards() {
        const n = this.predictedCache.shape[0];

        return this.predictedCache.sub(this.expectedCache).mul(2 / n);
    }
}
