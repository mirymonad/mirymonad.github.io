import { ReverseLossFunction } from "./functions/loss/loss.js";
import { ReverseFunction } from "./functions/reverse/reverse.js";

export class Network {
    
    /**
     * 
     * @param {ReverseLossFunction} loss
     * @param {ReverseFunction[]} composites 
     */
    static sequential(loss, ...composites) {
        composites[0].composite = loss;

        //
        for (let i = 0; i < composites.length-1; i++) {
            composites[i+1].composite = composites[i];
        }

        return composites.at(-1);
    }
}