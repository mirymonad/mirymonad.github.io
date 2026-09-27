import { NumTensor } from "../../tensor.js";
import { ReverseLossFunction } from "../loss/loss.js";

class ReverseFunction {

    /**
     * The write-only activation function
     * 
     * @type {ReverseFunction | ReverseLossFunction}
     */
    #composite;

    set composite(composite) {
        this.#composite = composite;
    }

    /**
     * The read-only forwards cache
     * 
     * @type {NumTensor}
     */
    #forwardsCache;

    get forwardsCache() {
        return this.#forwardsCache;
    }

    /**
     * Initialises method hooks to wrap default
     * logic that executes when they are called
     */
    constructor() {
        const step = this.step;

        // The step method hook
        this.step = (optimiser) => {
            step?.call(this, optimiser);
            this.#composite?.step?.(optimiser);
        }

        const forwards = this.forwards;

        // The forwards method hook
        this.forwards = (tensor) => {
            this.#forwardsCache = tensor;
            return forwards.call(this, tensor);
        }
    }

    /**
     * Computes the function application
     * 
     * @param {NumTensor} tensor the input tensor
     * @returns {NumTensor} the function application
     */
    forwards(tensor) {
        return this.#composite.forwards(tensor);
    }

    /**
     * Computes the recursive gradient
     * 
     * @returns {NumTensor} the recursive gradient
     */
    backwards() {
        return this.#composite.backwards();
    }
}

class WeightedReverseFunction extends ReverseFunction {

    /**
     * The perceptron weights
     * 
     * @type {NumTensor}
     */ 
    #weights;

    get weights() {
        return this.#weights;
    }

    set weights(weights) {
        this.#weights = weights;
    }

    /**
     * The read-only backwards cache
     * 
     * @type {NumTensor}
     */
    #backwardsCache;

    get backwardsCache() {
        return this.#backwardsCache;
    }

    constructor(inFeatures, outFeatures) {
        super();

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        // The neurons are the rows, the inputs are the columns
        this.#weights = new NumTensor([outFeatures, inFeatures]);
    }

    /**
     * Updates the weights using the optimiser
     * 
     * @param {any} optimiser the optimiser
     */
    step(optimiser) {
        this.weights = optimiser.optimise(this.weights, this.backwardsCache);
    }

    backwards() {
        const gradient = super.backwards();

        // Compute the weight gradients of the function
        this.#backwardsCache = this.cacheBackwards(gradient);

        // Return the recursive gradient for the caller
        return gradient;
    }

    /**
     * Computes weight gradients and caches them
     * 
     * @param {NumTensor} gradient the recursive gradient
     */
    cacheBackwards(gradient) {
        throw new Error("cacheBackwards() must be implemented by subclass");
    }

    /**
     * He weight initialisation.
     * 
     */
    he() {
        for (let i = 0; i < this.#weights._data.length; i++) {
            const u1 = Math.random();
            const u2 = Math.random();

            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

            this.#weights._data[i] = z * Math.sqrt(2 / this.inFeatures);
        }

        console.log(this.#weights);
    }
}

export { ReverseFunction, WeightedReverseFunction };