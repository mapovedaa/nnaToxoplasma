/*
 * optimizer.h
 *
 *  Created on: 22/04/2021
 *      Author: user
 */

#ifndef OPTIMIZER_OPTIMIZER_H_
#define OPTIMIZER_OPTIMIZER_H_

#pragma once

#include "../dataset/dataset.h"
#include "../neural/neuralnetwork.h"


class Optimizer
{
public:
	Optimizer();
	~Optimizer();

	virtual void minimize();

	void setNeuralNetwork(NeuralNetwork* net);

	void setDataset(Dataset* dataset);

	double getScore(Datatype d, int limit = -1);

	void minimizeThread();

protected:
	NeuralNetwork* _n;

	Dataset* _d;

};





#endif /* OPTIMIZER_OPTIMIZER_H_ */
