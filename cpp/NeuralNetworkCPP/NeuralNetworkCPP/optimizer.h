#ifndef OPTIMIZER_OPTIMIZER_H_
	#define OPTIMIZER_OPTIMIZER_H_
	
	#pragma once
	#include "dataset.h"
	#include "neuralnetwork.h"

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
