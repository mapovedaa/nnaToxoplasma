/*
 * dataset.h
 *
 *  Created on: 22/04/2021
 *      Author: user
 */

#ifndef DATASET_DATASET_H_
#define DATASET_DATASET_H_

#pragma once

#include <vector>
#include<string>

using namespace std;

enum Datatype {
	TRAIN,
	TEST
};

class Dataset
{
public:

	Dataset(string filename);
	~Dataset();

	const vector<const vector<double>*>& getIns(Datatype d) const;

	const vector<const vector<double>*>& getOuts(Datatype d) const;

	void split(double ptrain);

private:
	vector<vector<double>> _ins;
	vector<vector<double>> _outs;

	vector<const vector<double>*> _train_ins;
	vector<const vector<double>*> _train_outs;

	vector<const vector<double>*> _test_ins;
	vector<const vector<double>*> _test_outs;
};





#endif /* DATASET_DATASET_H_ */
