#include <iostream>
#include <stdio.h>					// printf, scanf, puts, NULL 
#include <string>					// string
#include <stdlib.h>					// srand, rand
#include <fstream>					// read write to file
#include <cmath>					// sqrt and pow

using namespace std;

/*---------------------------------------------*/
/* Declaration of Functions					   */
/*---------------------------------------------*/
void intro(double training_rate, int training_loops, int training_pairs, int input_neurons, int midlayer_neurons, int output_neurons);
void read_txt_file(int training_pairs, int input_neurons, double full_input[]);
void set_initial_weights(double layer_in_mid[], double layer_mid_out[], int input_neurons, int midlayer_neurons, int output_neurons);
void reset_neuron_states(double input_neuron_states[][3], double mid_neuron_states[][3], double out_neuron_states[][3], double target_values[]);
void activation(double neuron_states[][3], int count);
void feed_forward(int input_neurons, int midlayer_neurons, int output_neurons, double layer_in_mid[], double layer_mid_out[], double input_neuron_states[][3], double mid_neuron_states[][3], double out_neuron_states[][3]);
void back_propagation(int input_neurons, int midlayer_neurons, int output_neurons, double layer_in_mid[], double layer_mid_out[], double input_neuron_states[][3], double mid_neuron_states[][3], double out_neuron_states[][3], double training_rate);
void outro(int training_loops, int input_neurons, int midlayer_neurons, int output_neurons, double layer_in_mid[], double layer_mid_out[]);
void to_matlab(int training_loops, int input_neurons, int midlayer_neurons, int output_neurons, double layer_in_mid[], double layer_mid_out[]);

int main() {
	/*---------------------------------------------*/
	/* Variables and constants					   */
	/*---------------------------------------------*/
	const int N = 1;								// training rate
	const int nTloops = 1000;						// number of training loops
	const int nPairs = 4;							// number of training pairs
	
	const int nNeurIn = 35;							// no. of input neurons
	const int nNeurMid = 8;							// no. of hidden layer neurons
	const int nNeurOut = 4;							// no. of output neurons
	
	double ErrLog[nNeurOut + 1][nTloops * nPairs] = { 0 }; // error log

	ofstream myfile;								// ouptut file
	myfile.open("H:\\Visual Studio 2015\\Projects\\cpp_ANN\\cpp_ANN\\Output_Folder\\errorlog.txt", ios::trunc);

	double layer1w[nNeurIn * nNeurMid] = { 0 };		// array to store weights in 1st layer (input to midlayer neurons)
	double layer2w[nNeurMid * nNeurOut] = { 0 };	// array to store weights in 2nd layer (midlayer to output neurons)
	/*
	Array numbering described herein:
	layer1w[0][0] = Weight Input1 to N1		layer2w[0][0] = Weight N1 to Output1
	layer1w[1][0] = Weight Input2 to N1     layer2w[1][0] = Weight N2 to Output1
	layer1w[2][0] = Weight Input3 to N1     layer2w[2][0] = Weight N3 to Output1
	layer1w[3][0] = Weight Input1 to N2     layer2w[3][0] = Weight N1 to Output2
	layer1w[4][0] = Weight Input2 to N2     layer2w[4][0] = Weight N2 to Output2
	layer1w[5][0] = Weight Input3 to N2     layer2w[5][0] = Weight N3 to Output2
	etc.
	*/
	
	double input_read_array[nNeurIn * nPairs] = { 0 };
	double NeurIn[nNeurIn][3] = { 0 };				// arrays to store iterative state of each neuron (input layer) (0:input|1:output|2:error)
	double NeurMid[nNeurMid][3] = { 0 };			// arrays to store iterative state of each neuron (mid layer) (0:input|1:output|2:error)
	double NeurOut[nNeurOut][3] = { 0 };			// arrays to store iterative state of each neuron (output layer) (0:input|1:output|2:error)
	double Target[nNeurOut] = { 0 };				// arrays to store target state of each output neuron
	
	double init_total_error_sq = 0.00;
	double fin_total_error_sq = 0.00;
	

	/*---------------------------------------------*/
	/* Main										   */
	/*---------------------------------------------*/
	intro(N, nTloops, nPairs, nNeurIn, nNeurMid, nNeurOut);					// introduces the program in output window
	set_initial_weights(layer1w, layer2w, nNeurIn, nNeurMid, nNeurOut);		// set initial weights to random numbers
	read_txt_file(nPairs, nNeurIn, input_read_array);						// read all inputs into a single array

	cout << "\nStarting ANN Training..." << endl;
	for (int Tloop = 0; Tloop < nTloops; Tloop++) {							// iterates through training loops
		
		memset(NeurIn, 0, sizeof(NeurIn));									// Resets all neuron states
		memset(NeurMid, 0, sizeof(NeurMid));
		memset(NeurOut, 0, sizeof(NeurOut));
		memset(Target, 0, sizeof(Target));
		
		// Defining Training Pairs
		for (int Pair = 1; Pair <= nPairs; Pair++) {						// iterates through each training pair
			if (Pair == 1) {			// first training pair
				
				for (int c = 0; c < nNeurIn; c++) {
					NeurIn[c][1] = input_read_array[c + nPairs * (Pair - 1)];
				}

				for (int d = 0; d < nNeurOut; d++) {
					if (d == Pair - 1) {
						Target[d] = 1;
					}
					else {
						Target[d] = 0;
					}
				}
			}
			else if (Pair == 2) {		// second training pair
				
				for (int c = 0; c < nNeurIn; c++) {
					NeurIn[c][1] = input_read_array[c + nPairs * (Pair - 1)];
				}

				for (int d = 0; d < nNeurOut; d++) {
					if (d == Pair - 1) {
						Target[d] = 1;
					}
					else {
						Target[d] = 0;
					}
				}
			}
			else if (Pair == 3) {		// third training pair
				
				for (int c = 0; c < nNeurIn; c++) {
					NeurIn[c][1] = input_read_array[c + nPairs * (Pair - 1)];
				}
				
				for (int d = 0; d < nNeurOut; d++) {
					if (d == Pair - 1) {
						Target[d] = 1;
					}
					else {
						Target[d] = 0;
					}
				}
			}
			else if (Pair == 4) {		// fourth training pair
				
				for (int c = 0; c < nNeurIn; c++) {
					NeurIn[c][1] = input_read_array[c + nPairs * (Pair - 1)];
				}
				
				for (int d = 0; d < nNeurOut; d++) {
					if (d == Pair - 1) {
						Target[d] = 1;
					}
					else {
						Target[d] = 0;
					}
				}
			}
			else {						// error check
				cout << "Error: Training Pair not defined" << endl;
			}

			// feed forward (forward pass through ANN)
			
			feed_forward(nNeurIn, nNeurMid, nNeurOut, layer1w, layer2w, NeurIn, NeurMid, NeurOut);
						
			//Calculates output error and populates error log
			ErrLog[0][((Tloop * nPairs) + Pair - 1)] = ((Tloop * nPairs) + Pair - 1);
			for (int o = 0; o < nNeurOut; o++) {
				double squish = ((1 - NeurOut[o][1]) * NeurOut[o][1]);
				NeurOut[o][2] = (Target[o] - NeurOut[o][1]) * squish;
				ErrLog[o + 1][((Tloop * nPairs) + Pair - 1)] = NeurOut[o][2];
				
				if (Tloop == 0) {
					init_total_error_sq += pow(NeurOut[o][2],2);
				}

				if (Tloop == nTloops-1) {
					fin_total_error_sq += pow(NeurOut[o][2], 2);
				}
				
			}
			
			// Back Propagation Algorithm
			back_propagation(nNeurIn, nNeurMid, nNeurOut, layer1w, layer2w, NeurIn, NeurMid, NeurOut, N);
		}
	}
	
	
	cout << "The initial magnitude of all errors is: " << init_total_error_sq << endl;
	
	

	/*---------------------------------------------*/
	/* Outro and Logging Error					   */
	/*---------------------------------------------*/
	outro(nTloops, nNeurIn, nNeurMid, nNeurOut, layer1w, layer2w);
	
	cout << "\nNote: The final magnitude of all errors is: " << fin_total_error_sq << endl;

	// write error log to file and close file
	for (int i = 0; i < nTloops * nPairs; i++) {
		for (int j = 0; j <= nNeurOut; j++) {
			if (j < nNeurOut) {
				myfile << ErrLog[j][i] << ' ';
			}
			else if (j = nNeurOut) {
				myfile << ErrLog[j][i];
			}
		}
		myfile << endl;
	}
	myfile.close();
	
	/*---------------------------------------------*/
	/* Output to Matlab for verification		   */
	/*---------------------------------------------*/
	to_matlab(nTloops, nNeurIn, nNeurMid, nNeurOut, layer1w, layer2w);

	
	string y;
	getline(cin, y);
	return 0;
}

/*---------------------------------------------*/
/* Functions								   */
/*---------------------------------------------*/
void intro(double training_rate, int training_loops, int training_pairs, int input_neurons, int midlayer_neurons, int output_neurons) {
	// Introduction to the program
	cout << "Artificial Neural Network Training Program\n" << endl;
	cout << "Training rate set to: " << training_rate << endl;
	cout << "Number of Training Loops: " << training_loops << endl;
	cout << "Number of Training Pairs: " << training_pairs << endl;
	cout << "\nThe structure of the ANN is as follows (in | hidden | out):" << endl;
	cout << input_neurons << "\t|\t" << midlayer_neurons << "\t|\t" << output_neurons << endl;
}
void read_txt_file(int training_pairs, int input_neurons, double full_input[]) {
	// read inputs from text file and populate input arrays
	ifstream inFile;
	inFile.open("ABCD.txt");
	if (inFile.is_open()) {
		cout << "\nInput dataset opened successfully...";
		while (!inFile.eof()) {
			for (int c = 0; c < input_neurons*training_pairs; c++) {
				inFile >> full_input[c];
				//cout << full_input[c]; // commented out
			}
			//cout << "\n\n" << endl; // commented out
		}
	}
	cout << "data read completed." << endl;
	inFile.close();
}
void set_initial_weights(double layer_in_mid[], double layer_mid_out[], int input_neurons, int midlayer_neurons, int output_neurons) {
	// populates weights arrays with random numbers ( -1 < num < +1 )
	ofstream initial_weights;
	initial_weights.open("H:\\Visual Studio 2015\\Projects\\cpp_ANN\\cpp_ANN\\Output_Folder\\initial_weights.txt", ios::trunc);
	
	
	initial_weights << "The initial weights in layer 1 are:" << endl;
	for (int i = 0; i < (input_neurons * midlayer_neurons); i++) {
		layer_in_mid[i] = (rand() % 201 - 100) / 100.00;
		initial_weights << layer_in_mid[i] << endl;
	}
	initial_weights << "\nThe initial weights in layer 2 are:" << endl;
	for (int i = 0; i < (midlayer_neurons * output_neurons); i++) {
		layer_mid_out[i] = (rand() % 201 - 100) / 100.00;
		initial_weights << layer_mid_out[i] << endl;
	}
	initial_weights.close();
}
void reset_neuron_states(double input_neuron_states[][3], double mid_neuron_states[][3], double out_neuron_states[][3], double target_values[]) {
	// resets all neuron states // note: not currently used.
	memset(input_neuron_states, 0, sizeof(input_neuron_states));
	memset(mid_neuron_states, 0, sizeof(mid_neuron_states));
	memset(out_neuron_states, 0, sizeof(out_neuron_states));
	memset(target_values, 0, sizeof(target_values));
}
void activation(double neuron_states[][3], int count) {
	neuron_states[count][1] = 1 / (1 + exp(-neuron_states[count][0]));
}
void feed_forward(int input_neurons, int midlayer_neurons, int output_neurons, double layer_in_mid[], double layer_mid_out[], double input_neuron_states[][3], double mid_neuron_states[][3], double out_neuron_states[][3]) {
	// Feed forward through the ANN
	for (int m = 0; m < midlayer_neurons; m++) { // Calculates the Inputs to each mid layer neuron
		for (int i = 0; i < input_neurons; i++) {
			int rep = m * input_neurons + i;
			mid_neuron_states[m][0] += input_neuron_states[i][1] * layer_in_mid[rep];
		}
	}
	for (int m = 0; m < midlayer_neurons; m++) { // Calculates the Outputs from each mid layer neuron
		activation(mid_neuron_states, m);
	}
	for (int o = 0; o < output_neurons; o++) { // Calculates the Inputs to each output layer neuron
		for (int m = 0; m < midlayer_neurons; m++) {
			int rep = o * midlayer_neurons + m;
			out_neuron_states[o][0] += mid_neuron_states[m][1] * layer_mid_out[rep];
		}
	}
	for (int o = 0; o < output_neurons; o++) { // Calculates the Outputs from each output layer neuron
		activation(out_neuron_states, o);
	}
}
void back_propagation(int input_neurons, int midlayer_neurons, int output_neurons, double layer_in_mid[], double layer_mid_out[], double input_neuron_states[][3], double mid_neuron_states[][3], double out_neuron_states[][3], double training_rate) {
	// back propagation algorithm
	for (int o = 0; o < output_neurons; o++) { // adjust layer 2 weights
		for (int m = 0; m < midlayer_neurons; m++) {
			int rep = (o * midlayer_neurons) + m;
			layer_mid_out[rep] += training_rate * out_neuron_states[o][2] * mid_neuron_states[m][1];
		}
	}
	for (int m = 0; m < midlayer_neurons; m++) { // calculate the error in the hidden layer neurons
		double squish = ((1 - mid_neuron_states[m][1]) * mid_neuron_states[m][1]);
		for (int o = 0; o < output_neurons; o++) {
			int rep = m * output_neurons + o;
			mid_neuron_states[m][2] += out_neuron_states[o][2] * layer_mid_out[rep] * squish;
		}
	}
	for (int m = 0; m < midlayer_neurons; m++) { // adjust layer 1 weights
		for (int i = 0; i < input_neurons; i++) {
			int rep = (m * input_neurons) + i;
			layer_in_mid[rep] += training_rate * mid_neuron_states[m][2] * input_neuron_states[i][1];
		}
	}
}
void outro(int training_loops, int input_neurons, int midlayer_neurons, int output_neurons, double layer_in_mid[], double layer_mid_out[]) {
	// outputs sent to output file
		
	ofstream final_weights;
	final_weights.open("H:\\Visual Studio 2015\\Projects\\cpp_ANN\\cpp_ANN\\Output_Folder\\final_weights.txt", ios::trunc);
	
	cout << "ANN weights have been adjusted following " << training_loops << " training loops." << endl;
	//cout << "The final avr loss function magnitude is: " << final_loss << endl;
	cout << "...ANN Training complete." << endl;

	for (int i = 0; i < (input_neurons * midlayer_neurons); i++) {
		final_weights << layer_in_mid[i] << endl;
	}
	final_weights << "\nThe final adjusted weights in layer 2 are:" << endl;
	for (int i = 0; i < (midlayer_neurons * output_neurons); i++) {
		final_weights << layer_mid_out[i] << endl;
	}
	final_weights.close();
	cout << "\nOutput files written into project directory:" << endl;
	cout << ".\\Projects\\cpp_ANN\\cpp_ANN\\Output_Folder\\..." << endl;
	cout << "\n-----END------" << endl;
}
void to_matlab(int training_loops, int input_neurons, int midlayer_neurons, int output_neurons, double layer_in_mid[], double layer_mid_out[]) {
	// outputs sent to txt file
	ofstream matlab_outfile;								// ouptut file
	matlab_outfile.open("H:\\Visual Studio 2015\\Projects\\cpp_ANN\\cpp_ANN\\Output_Folder\\to_matlab.txt", ios::trunc);


	matlab_outfile << "The following should be copied into MATLAB 'ANN_verifier' for testing:\n" << endl;
	matlab_outfile << "nNeurIn = " << input_neurons << ";" << endl;
	matlab_outfile << "nNeurMid = " << midlayer_neurons << ";" << endl;
	matlab_outfile << "nNeurOut = " << output_neurons << ";" << endl;
	matlab_outfile << "layer1w = [";
	for (int i = 0; i < (input_neurons * midlayer_neurons); i++) {
		if (i < (input_neurons * midlayer_neurons - 1)) {
			matlab_outfile << layer_in_mid[i] << "; ";
		}
		else {
			matlab_outfile << layer_in_mid[i];
		}
	}
	matlab_outfile << "];" << endl;
	matlab_outfile << "layer2w = [";
	for (int i = 0; i < (midlayer_neurons * output_neurons); i++) {
		if (i < (midlayer_neurons * output_neurons - 1)) {
			matlab_outfile << layer_mid_out[i] << "; ";
		}
		else {
			matlab_outfile << layer_mid_out[i];
		}
	}
	matlab_outfile << "];" << endl;
	matlab_outfile.close();
}
