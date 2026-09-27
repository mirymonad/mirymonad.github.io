class LossFunction {

    /**
     * 
     */
    #predictedCache;

    get predictedCache() {
        return this.#predictedCache;
    }

    /**
     * 
     */
    #expectedCache;

    get expectedCache() {
        return this.#expectedCache;
    }

    constructor() {
        const apply = this.apply;

        // The apply method hook
        this.apply = (predicted, expected) => {
            this.#predictedCache = predicted;
            this.#expectedCache = expected;

            return apply.call(this, predicted, expected);
        }
    }

    apply() {
        throw new Error("apply() must be implemented by subclass");
    }
}

/**
 * A loss function that implements the reverse function composite interface.
 */
class ReverseLossFunction extends LossFunction {

    forwards(tensor) {
        return tensor;
    }

    backwards() {
        throw new Error("backwards() must be implemented by subclass");
    }
}

export { LossFunction, ReverseLossFunction };