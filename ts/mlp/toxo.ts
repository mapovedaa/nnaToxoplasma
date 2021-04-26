import * as Chartjs from 'chart.js';
import * as ChartjsNode from 'chartjs-node-canvas';
import { BackPropagation } from './back-propagation';
import { TransferFunction } from './transfer-function';

import * as fs from 'fs';

const readFile = (dataset: string, result: string) => {
    let input: number[][] = [];
    let output: number[][] = [];

    const fileData = fs.readFileSync(dataset).toString();
    const fileResultData = fs.readFileSync(result).toString();
    const data = fileData.split('\n');
    const resultData = fileResultData.split('\n');


    let features: any = data.shift()?.split(',');

    data.forEach((line: any) => {
        if (line != '')
            input.push(line.substr(line.indexOf(',') + 1).replace('\r', '').split(',').map((item: string) => {
                return parseFloat(item);
            }));
    });


    resultData.forEach((line: any) => {
        if (line != '')
            output.push(line.substr(line.indexOf(',') + 1).replace('\r', '').split(',').map((item: string) => {
                return parseFloat(item);
            }));
    });

    return [input, output];
};

const trainingData = readFile('./../data/training.csv',
    './../data/result_training.csv');

let input: number[][] = trainingData[0];
let output: number[][] = trainingData[1];

let valuesMean: any[] = [];
console.time('mlp-toxo');

for (var index = 0; index < 30; index++) {
    let values: number[] = [];

    let network = new BackPropagation(
        [input[0].length, 45, 12, 6, 3, 1],
        [TransferFunction.NONE, TransferFunction.LINEAR, TransferFunction.LINEAR,
        TransferFunction.LINEAR, TransferFunction.LINEAR, TransferFunction.SIGMOID]
    );

    const maxCount = 500;
    const size = input.length;

    let error = 0.0;
    let count = 0;

    do {
        count++;
        error = 0.0;

        for (var i = 0; i < size; i++) {
            error += network.train(input[i], output[i], 0.7, 0.1);
        }

        error = error / size;
        // Show progress
        values.push(error);

        if (count % 100 === 0) {
            console.log(`Epoch ${count} completed with error ${error}`);
        }
    } while (error > 0.25 && count <= maxCount);

    let pathFileResult: string = './output-toxo/result-' + index + '.txt';
    fs.writeFile(pathFileResult,
        values.toString(),
        (err: any) => {
            if (err)
                console.log(err);
            else {
                console.log('File written successfully\n');
            }
        });
}

console.timeEnd('mlp-toxo')
