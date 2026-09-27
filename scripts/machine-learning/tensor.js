/**
 * Swap and copy
 * 
 * @param {*} array 
 * @param {*} i 
 * @param {*} j 
 * @returns 
 */
function swappy(array, i, j) {
    const copy = [...array];

    copy[i] = array[j];
    copy[j] = array[i];

    return copy;
}

class Tensor {
    /**
     * An array containing the sizes of the contiguous row-major order blocks
     * of each dimensions at the corresponding shape index. 
     * 
     * For example, given a tensor with shape `(1, 2, 3, 4)`:
     * 
     * ```
     * _strides.at(3) = 1 // a single cell
     * _strides.at(2) = 4 // the end block
     * _strides.at(1) = 3 * 4 // the last two blocks
     * _strides.at(0) = 2 * 3 * 4 // the last three blocks
     * ```
     * 
     *  @type {number[]}
     */
    _strides;

    /**
     * A float array containing the contigious row-major order tensor values.
     * 
     *  @type {Float32Array}
     */
     _data;

    get min() {
        return Math.min(...this._data);
    }

    get max() {
        return Math.max(...this._data);
    }

    /**
     * Initialises a new tensor instance.
     * 
     * @param {number[]} shape the tensor shape.
     * @param {Float32Array} data the tensor data.
     */
    constructor(shape, data = null) {
        this.shape = shape.length ? shape : (shape = [1]);

        // Initialise the 1D strides array
        this._strides = new Array(shape.length+1).fill(1);
        this._strides[shape.length-1] = shape[shape.length-1];

        for (let i = this._strides.length-3; i >= 0; i--) {
            this._strides[i] = this._strides[i+1] * shape[i];
        }

        // Initialise the 1D float32 array
        const length = this._strides.shift();

        if (data && data.length !== length) {
            throw new Error("invalid input data");
        }

        this._data = data ?? new Float32Array(length);
    }

    /**
     * Returns the contiguous row-major order block at the given indices.
     * 
     * @param {number[]} indices the indices.
     * @returns the block at the indices.
     */
    at(indices) {
        let index = 0;

        // Calculate the index from _strides
        for (let i = 0; i < indices.length; i++) {
            index += this._strides[i] * indices[i];
        }

        // Retrieve the column to be indexed
        return this._data.slice(index, index + this._strides[indices.length-1]);
    }

    /**
     * Converts a current tensor indices into an index.
     * 
     * @see {@link unravel} for the opposite.
     * @param {number[]} indices the indices.
     * @returns the ravelled index.
     */
    ravel(indices) {
        let index = 0;

        // Calculate the index from _strides
        for (let i = 0; i < indices.length; i++) {
            index += this._strides[i] * indices[i];
        }

        return index;
    }
    
    /**
     * Converts a current tensor index into an indices.
     * 
     * @see {@link ravel} for the opposite.
     * @param {number} index the index.
     * @returns the unravelled indices.
     */
    unravel(index) {
        let indices = []

        // Calculate each indices from _strides
        for (let i = 0; i < this.shape.length; i++) {
            indices.push(Math.floor(index / this._strides[i]));
            index %= this._strides[i];
        }

        return indices;
    }
}

class NumTensor extends Tensor {

    constructor(shape, data = null) {
        super(shape, data);

        if (!data) {
            this._data.fill(0);
        }
    }

    /**
     * Permutes the tensor at the provided dimensions.
     * 
     * Specifically, given the indexes of two dimensions in the shape of the
     * current tensor, each values indices are unraveled before being swapped
     * at the provided dimensions, with the value then being inserted at the
     * newly constructed indices in the resulting tensor. 
     * 
     * For example, given the tensor:
     * 
     * ```
     * let A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
     * ```
     * 
     * Then `A.permute(0, 1)` would perform:
     * 
     * ```
     * C = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
     * ```
     * 
     * This performs matrix transpositon by swapping each row with each column
     * when the current tensor is two-dimensional.
     * 
     * @note the order of the input parameters does not matter.
     * @param {number} dim1 dimension index one.
     * @param {number} dim2 dimension index two.
     * @returns the tensor permutation.
     */
    permute(dim1, dim2) {
        const result = new NumTensor(swappy(this.shape, dim1, dim2));

        for (let i = 0; i < this._data.length; i++) {
            const indices = this.unravel(i);

            [indices[dim1], indices[dim2]] = [indices[dim2], indices[dim1]];

            // Set at the result array
            let index = 0;

            // Calculate the index from _strides
            for (let i = 0; i < indices.length; i++) {
                index += result._strides[i] * indices[i];
            }

            result._data[index] = this._data[i];
        }

        return result;
    }

    get average() {
        return Math.floor(this._data.reduce((x, y) => x + y) / this._data.length);
    }

    /**
     * Adds each value from the input tensor.
     * 
     * For example, given two tensors:
     * 
     * ```
     * let A = [1, 2, 3]
     * let B = [4, 5, 6]
     * ```
     * 
     * Then `A.sub(B)` would perform:
     * 
     * ```
     * C = [1+4, 2+5, 3+6]
     * C = [5, 7, 9]
     * ```
     * 
     * @param {NumTensor} tensor the input tensor.
     * @returns the tensor addition.
     */
    add(tensor) {
        const result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] + tensor._data[i];
        }

        return result;
    }

    #add_fast(array1, array2) {

        for (let i = 0; i < array1.length; i++) {
            array1[i] += array2[i];
        }

        return array1;
    }

    /**
     * Subtracts each value from the input tensor.
     * 
     * For example, given two tensors:
     * 
     * ```
     * let A = [1, 2, 3]
     * let B = [4, 5, 6]
     * ```
     * 
     * Then `A.sub(B)` would perform:
     * 
     * ```
     * C = [1-4, 2-5, 3-6]
     * C = [-3, -3, -3]
     * ```
     * 
     * @param {NumTensor} tensor the input tensor.
     * @returns the tensor subtraction.
     */
    sub(tensor) {
        const result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] - tensor._data[i];
        }

        return result;
    }

    /**
     * Performs elementwise multiplication.
     * 
     * For example, given the tensor:
     * 
     * ```
     * let A = [1, 2, 3]
     * ```
     * 
     * Then `A.mul(2)` would perform:
     * 
     * ```
     * C = [1*2, 2*2, 3*2]
     * C = [2, 4, 6]
     * ```
     * 
     * @param {number} scalar the number to multiply elementwise.
     * @returns the tensor multiplication.
     */
    mul(scalar) {
        let result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] * scalar;
        }

        return result;
    }

    #mul_fast(array, scalar) {
        
        for (let i = 0; i < array.length; i++) {
            array[i] *= scalar;
        }

        return array;
    }

    /**
     * Performs elementwise division.
     * 
     * For example, given the tensor:
     * 
     * ```
     * let A = [1, 2, 3]
     * ```
     * 
     * Then `A.div(2)` would perform:
     * 
     * ```
     * C = [1/2, 2/2, 3/2]
     * C = [0.5, 1, 1.5]
     * ```
     * 
     * @param {number} scalar the number to divide elementwise.
     * @returns the tensor division.
     */
    div(scalar) {
        let result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] / scalar;
        }

        return result;
    }

    /**
     * Performs an elementwise power.
     * 
     * For example, given the tensor:
     * 
     * ```
     * let A = [1, 2, 3]
     * ```
     * 
     * Then `A.pow(2)` would perform:
     * 
     * ```
     * C = [1^2, 2^2, 3^2]
     * C = [1, 4, 9]
     * ```
     * 
     * @param {number} scalar the power to apply elementwise.
     * @returns the power scaled tensor.
     */
    pow(scalar) {
        const result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = Math.pow(this._data[i], scalar);
        }

        return result;
    }

    /**
     * Computes the hadamard product.
     * 
     * Specifically, assuming the current tensor and input tensor both have the
     * same shape, each value at the same indices in both tensors is multiplied
     * together to form the resulting tensor. 
     * 
     * For example, given two tensors:
     * 
     * ```
     * let A = [1, 2, 3]
     * let B = [4, 5, 6]
     * ```
     * 
     * Then `A.hadamard(B)` would perform:
     * 
     * ```
     * C = [1*4, 2*5, 3*6]
     * C = [4, 10, 18]
     * ```
     * 
     * @param {NumTensor} tensor the input tensor.
     * @returns the hadamard product.
     */
    hadamard(tensor) {
        let result = new NumTensor([this._data.length]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] * tensor._data[i];
        }

        return result;
    }

    /**
     * Computes a row-vector product.
     * 
     * Specifically, each starting-row in the current tensor of shape (a, ...)
     * is multiplied by the input vector value (a_i) at the same corresponding
     * index, before every resulting sub-tensor with shape (...) is summed.
     * 
     * For example, given the tensor:
     * 
     * ```
     * let A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
     * ```
     * 
     * And the vector:
     * 
     * ```
     * let v = [1, 2, 3]
     * ```
     * 
     * And the result:
     * 
     * ```
     * let C = [0, 0, 0]
     * ```
     * 
     * Then `A.v_row_mul(v)` would perform:
     * 
     * ```
     * C += A.at([0]).mul(v[0])
     * C += A.at([1]).mul(v[1])
     * C += A.at([2]).mul(v[2])
     * ```
     * 
     * Giving a resulting tensor `C` of:
     * 
     * ```
     * C = [30, 36, 42]
     * ```
     * 
     * @param {number[]} vector the input vector. 
     * @returns the row-vector product.
     */
    v_row_mul(vector) {
        let result = new NumTensor(this.shape.slice(1));

        for (let i = 0; i < this.shape[0]; i++) {
            result.#add_fast(result._data, this.#mul_fast(this.at([i]), vector[i]));
        }

        return result;
    }

    /**
     * Computes a tensor contraction.
     * 
     * Specifically, it multiplies each end-row in the current tensor against
     * each starting-row in the input tensor. The shape of the current tensor
     * columns and the input tensors rows (..., a) (a, ...) hence must match
     * to ensure an equal amount of rows are multiplied.
     *  
     * For example, given two tensors:
     * 
     * ```
     * let A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
     * let B = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
     * ```
     * 
     * Then, `A.t_mul(B)` would perform:
     * 
     * ```
     * C_r1 = B.v_row_mul([1, 2, 3])
     * C_r2 = B.v_row_mul([4, 5, 6])
     * C_r3 = B.v_row_mul([7, 8, 9])
     * ```
     * 
     * Giving a resulting tensor `C` of:
     * 
     * ```
     * C = [[84, 90, 96], [201, 216, 231], [318, 342, 366]]
     * ```
     * 
     * This performs matrix multiplication by treating each row in matrix A as
     * a row-vector applied to a vector-matrix product with matrix B.
     * 
     * @param {NumTensor} tensor the input tensor.
     * @returns the tensor contraction.
     */
    t_mul(tensor) {
        const result = new NumTensor([...this.shape.slice(0, -1), ...tensor.shape.slice(1)]);

        // Compute the corresponding tensor row sizes to multiply
        const iStep = this._strides.at(-1) * this.shape.at(-1);
        const jStep = result._strides.at(-tensor.shape.length);
        
        for (let i = 0, j = 0; i < this._data.length;) {
            const vectorData = this._data.subarray(i, (i += iStep));
            const resultData = result._data.subarray(j, (j += jStep));
            this.#add_fast(resultData, tensor.v_row_mul(vectorData)._data);
        }

        return result;
    }

    /**
     * Computes the outer product.
     * 
     * Specifically, it multiplies each value in the current tensor against the
     * entire input tensor, and inserts the resulting tensor at the same indices
     * as the value from the current tensor. This transforms each value in the
     * current tensor into a sub-tensor of input tensor shape.
     * 
     * For example, given two tensors:
     * 
     * ```
     * let A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
     * let B = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
     * ```
     * 
     * Then, `A.outer(B)` would perform:
     * 
     * ```
     * C_t1 = 1 * B
     * C_t2 = 2 * B
     * C_t3 = 3 * B
     * C_t4 = 4 * B
     * C_t5 = 5 * B
     * C_t6 = 6 * B
     * C_t7 = 7 * B
     * C_t8 = 8 * B
     * C_t9 = 9 * B
     * ```
     * 
     * Giving a resulting tensor `C` of:
     * 
     * ```
     * C = 
     * [
     *   [
     *     [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
     *     [[20, 22, 24], [26, 28, 30], [32, 34, 36]],
     *     [[30, 33, 36], [39, 42, 45], [48, 51, 54]]
     *   ],
     *   [
     *     [[40, 44, 48], [52, 56, 60], [64, 68, 72]],
     *     [[50, 55, 60], [65, 70, 75], [80, 85, 90]],
     *     [[60, 66, 72], [78, 84, 90], [96, 102, 108]]
     *   ],
     *   [
     *     [[70, 77, 84], [91, 98, 105], [112, 119, 126]],
     *     [[80, 88, 96], [104, 112, 120], [128, 136, 144]],
     *     [[90, 99, 108], [117, 126, 135], [144, 153, 162]]
     *   ]
     * ]
     * ```
     * 
     * This performs the outer product, as each values indices in the current
     * tensor becomes the indices for a sub-tensor row in the resulting tensor.
     * Therefore, the current tensor values are multiplied across the rows and
     * input tensor values are multiplied across the columns. 
     * 
     * @param {NumTensor} tensor the input tensor.
     * @returns the outer product.
     */
    outer(tensor) {
        const result = new NumTensor([...this.shape, ...tensor.shape]);
        
        for (let i = 0, j = 0; i < this._data.length; i++) {
            const resultData = result._data.subarray(j, (j += tensor._data.length));
            this.#add_fast(resultData, this.#mul_fast([...tensor._data], this._data[i]));
        }

        return result;
    }
}

export { Tensor, NumTensor };