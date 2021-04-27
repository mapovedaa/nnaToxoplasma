#ifndef MISC_FUNCTIONS_H_
	#define MISC_FUNCTIONS_H_
	#ifndef FUNCTIONS_H
		#define FUNCTIONS_H
		
		#include <iostream>
		#include <vector>
		#include <utility>
		#include <limits>
		#include <string>
		#include <math.h>
		#include <stdio.h>

		using namespace std;

		//Sigmoid Function
		double sigmoid(double x);
		double sigmoid_derivative(double x);

		//Relu Function
		double relu(double x);
		double relu_derivative(double x);

		//Random float getter function
		double random(double low, double high);
		
		double distanceVector(const vector<double>& v1, const vector<double>& v2);
	#endif // FUNCTIONS_H
#endif /* MISC_FUNCTIONS_H_ */
