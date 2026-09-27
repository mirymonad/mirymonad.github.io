import { NumTensor } from "../tensor";

class MinMax extends Scaler {

    /**
     * The minimum value of the unscaled dataset.
     * 
     * @type {number}
     */
    min;

    /**
     * The maximum value of the unscaled dataset.
     * 
     * @type {number}
     */
    max;

    /**
     * @param {NumTensor} tensor
     */
    scale(tensor) {
        this.min = tensor.min;
        this.max = tensor.max;

        // 
        return tensor.sub(this.min).div(this.max - this.min);
    }

    /**
     * 
     * @param {NumTensor} tensor 
     * @returns 
     */
    unscale(tensor) {
        return tensor.mul(this.max - this.min) + this.min;
    }
}