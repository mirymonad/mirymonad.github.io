import { NumTensor } from "../tensor.js";

export class DataLoader {

    /**
     * 
     * 
     * @type {number[][]}
     */
    #features

    /**
     * 
     * 
     * @type {number[]}
     */
    #labels

    /**
     * 
     * @param {NumTensor} features the 2D features tensor 
     * @param {NumTensor} labels the 1D labels tensor 
     */
    constructor(features, labels) {
        this.#features = features;
        this.#labels = labels;

        if (features.length < 1) {
            throw new Error("features array is empty");
        }

        // Verify features array is of uniform shape
        const featureSize = this.#features[0].length;

        if (!features.every(feature => feature.length === featureSize)) {
            throw new Error("features nested arrays are of different sizes");
        }

        if (labels.length < 1) {
            throw new Error("labels array is empty");
        }

        // Verify labels array is of uniform shape
        const labelSize = this.#labels[0].length;

        if (!labels.every(label => label.length === labelSize)) {
            throw new Error("labels nested arrays are of different sizes");
        }

        // Verify there's a label for every feature
        if (features.length !== labels.length) {
            throw new Error("features and labels sizes are different");
        }
    }

    *[Symbol.iterator]() {

        for (let i = 0; i < this.#features.length; i++) {
            const featureTensor = new NumTensor(
                [this.#features[i].length],
                new Float32Array(this.#features[i])
            );

            const labelTensor = new NumTensor(
                [this.#labels[i].length],
                new Float32Array(this.#labels[i])
            );
                
           yield [featureTensor, labelTensor];
        }
    }
}