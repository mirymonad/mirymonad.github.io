import { DuelTensor, Tensor } from "../../tensor.js";

class Duel {

    /**
     * @param {number} real
     * @param {number} duel
     */
    constructor(real, duel) {
        this.real = real;
        this.duel = duel;
    }

    add(other) {
        return this.#update(
            this.real + other.real,
            this.duel + other.duel
        );
    }

    mul(other) {
        return this.#update(
            this.real * other.real,
            this.duel * other.real + this.real * other.duel
        );
    }

    pow(exp) {
        return this.#update(
            Math.pow(this.real, exp),
            exp * Math.pow(this.real, exp - 1) * this.duel
        );
    }

    sin() {
        return this.#update(
            Math.sin(this.real),
            Math.cos(this.real) * this.duel
        );
    }

    cos() {
        return this.#update(
            Math.cos(this.real),
            -Math.sin(this.real) * this.duel
        );
    }

    tan() {
        return this.#update(
            Math.tan(this.real),
            (1 + Math.pow(Math.tan(this.real), 2)) * this.duel
        );
    }

    log() {
        return this.#update(
            Math.log(this.real),
            (1.0 / this.real) * this.duel
        );
    }

    #update(real, duel) {
        this.real = real;
        this.duel = duel;
        return this;
    }
}

class Variable {
    /** @type {Duel} */ duel;
    /** @type {string} */ name;

    constructor(name, duel = null) {
        this.name = name;
        this.duel = duel;
    }
}

class ForwardFunction {

    /**
     * @param {Array<Variable>} vars
     * @param {Array<number>} respect
     */
    constructor(vars, respect) {
        this.vars = vars;
        this.respect = respect;
        this.#validateRespect();
    }

    /**
     * @param {Array<number>} args 
     * @returns {Tensor}
     */
    apply(...args) {
        const tensor = new Array(args.length);

        if (this.vars.length !== args.length) {
            throw new Error("invalid input argument");
        }

        for (let i = 0; i < this.vars.length; i++) {
            this.vars[i].duel = tensor[i] = new Duel(args[i], this.respect[i]);
        }

        return new DuelTensor(tensor);
    }

    #validateRespect() {

        if (this.vars.length !== this.respect.length) {
            throw new Error("respect bitmap is wrong length");
        }

        for (let i = 0; i < this.respect.length; i++) {

            if (this.respect[i] !== 0 && this.respect[i] !== 1) {
                throw new Error("respect bitmap must only contain 1s and 0s");
            }
        }
    }
}

export { Duel, Variable, ForwardFunction };