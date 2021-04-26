/*
 * backpropagation.h
 *
 *  Created on: 22/04/2021
 *      Author: user
 */

#ifndef OPTIMIZER_BACKPROPAGATION_H_
#define OPTIMIZER_BACKPROPAGATION_H_

#pragma once

#include "../misc/functions.h"
#include "../neural/neuralnetwork.h"
#include "../dataset/dataset.h"
#include "optimizer.h"
#include <unordered_map>

extern double LEARNING_RATE;


class Backpropagation : public Optimizer
{

public:
	void setLearningRate(double lr);

	vector<vector<vector<double>>> getBackpropagationShifts(const vector<double>& in, const vector<double>& out);

	void backpropagate(const vector<const vector<double>*>& ins, const vector<const vector<double>*>& outs);

	vector<Layer*> getLayers();

	void minimize();

	void setBatchSize(size_t bs);

private:
	size_t _batch_size = 20;
};




















#endif /* OPTIMIZER_BACKPROPAGATION_H_ */
