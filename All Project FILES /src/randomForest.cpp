/*
Name       : Ravi Kant
USC ID     : 7945-0425-48	
e-mail     : rkant@usc.edu	
Submission : Dec 8, 2015

Input Format: programName Data_Location ClassA_name number_of_classA_samples
				classB_name number_of_classB_samples number_of_test_samples

 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <iterator>
#include <vector>
using namespace cv;
using namespace std;


#include <opencv/cv.h>       // opencv general include file
#include <opencv/ml.h>		  // opencv machine learning include file

// parameter to set
// This is eaual to the num_clusters used in SIFT_PCA_BagOfWords.cpp
int ATTRIBUTES_PER_SAMPLE = 300;

// parameters set from dataDecription.txt
int num_training_samples_class;
int num_testing_samples_class;
int NUMBER_OF_CLASSES = 10;
int NUMBER_OF_TRAINING_SAMPLES;
int NUMBER_OF_TESTING_SAMPLES ;


void normalize(Mat* srcMat, Mat* dstMat) {

	int nRows = srcMat->rows;
	int nCols = srcMat->cols;

	vector<float> maxValues(nCols), minValues(nCols);
	srcMat->row(0).copyTo(maxValues);
	srcMat->row(0).copyTo(minValues);
	for(int col = 0; col < nCols; col++) {
		float max = maxValues[col];
		float min = minValues[col];
		float temp;
		for(int row = 0; row < nRows; row++) {
			temp = dstMat->at<float>(row,col);
			if(temp > max)
				max = temp;
			if(temp < min)
				min = temp;
		}
		maxValues[col] = max;
		minValues[col] = min;
	}

	for(int row = 0; row < nRows; row++) {
		for(int col = 0; col < nCols; col++) {
			dstMat->at<float>(row,col) = float(srcMat->at<float>(row,col) - minValues[col])/float(maxValues[col] - minValues[col]);
		}
	}
	for(int col = 0; col < nCols; col++) {
		cout<<maxValues[col]<<" ";
	}
	cout<<"\n";
	for(int col = 0; col < nCols; col++) {
		cout<<minValues[col]<<" ";
	}
}


/******************************************************************************/

int main( int argc, char** argv )
{

	// read dataDescription to know number of training and testing samples
	ifstream fin;
	fin.open("dataDescription.txt");
	if(fin){
		string temp;
		getline(fin,temp);
		getline(fin,temp);
		num_training_samples_class = atoi(temp.c_str());
		getline(fin,temp);
		num_testing_samples_class = atoi(temp.c_str());
		fin.close();
	}
	else
	{
		cout<<"Unable to open dataDesciption.txt\n";
		exit(1);
	}
	NUMBER_OF_TRAINING_SAMPLES = num_training_samples_class * 10;
	NUMBER_OF_TESTING_SAMPLES = num_testing_samples_class * 10;

	// define training data storage matrices (one for attribute examples, one
	// for classifications)

	Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
	Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

	//define testing data storage matrices

	Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
	Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

	// define all the attributes as numerical
	// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
	// that can be assigned on a per attribute basis

	Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
	var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

	// this is a classification problem (i.e. predict a discrete number of class
	// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

	var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

	double result; // value returned from a prediction

	// load training and testing data sets

	//ifstream fin;
	fin.open("training_feature_histogram_images.txt");
	for(int row = 0; row < NUMBER_OF_TRAINING_SAMPLES; row++){

		string line;
		getline(fin,line);

		std::istringstream buf(line);
		std::istream_iterator<std::string> beg(buf), end;

		std::vector<std::string> tokens(beg, end); // done!

		for(int i = 0; i < ATTRIBUTES_PER_SAMPLE; i++){
			training_data.at<float>(row,i) = (atof(tokens[i].c_str()));
			//  cout<<training_data.at<float>(row,i)<<" ";
		}
		//	cout<<"\n";
	}
	fin.close();
	//exit(0);


	fin.open("training_label_histogram_images.txt");
	for(int row = 0; row < NUMBER_OF_TRAINING_SAMPLES; row++){

		string line;
		getline(fin,line);

		training_classifications.at<float>(row,1) = float(atoi(line.c_str()));
	//	cout<<training_classifications.at<float>(row,1)<<"\n";

		//	cout<<"\n";
	}
	fin.close();


	fin.open("testing_feature_histogram_images.txt");
	for(int row = 0; row < NUMBER_OF_TESTING_SAMPLES; row++){

		string line;
		getline(fin,line);

		std::istringstream buf(line);
		std::istream_iterator<std::string> beg(buf), end;

		std::vector<std::string> tokens(beg, end); // done!

		for(int i = 0; i < ATTRIBUTES_PER_SAMPLE; i++){
			testing_data.at<float>(row,i) = (atof(tokens[i].c_str()));
			//  cout<<training_data.at<float>(row,i)<<" ";
		}
		//	cout<<"\n";
	}
	fin.close();

	fin.open("testing_labels_histogram_images.txt");
	for(int row = 0; row < NUMBER_OF_TESTING_SAMPLES; row++){
		string line;
		getline(fin,line);
		testing_classifications.at<float>(row,1) = float(atoi(line.c_str()));
	//	cout<<testing_classifications.at<float>(row,1)<<"\n";

	}
	fin.close();


	//exit(0);

	if (1==1)
	{


		// define the parameters for training the random forest (trees)

		float priors[] = {1,1,1,1,1,1,1,1,1,1};  // weights of each classification for classes
		// (all equal as equal samples of each digit)

		CvRTParams params = CvRTParams(25, // max depth
				5, // min sample count
				0, // regression accuracy: N/A here
				false, // compute surrogate split, no missing data
				15, // max number of categories (use sub-optimal algorithm for larger numbers)
				priors, // the array of priors
				false,  // calculate variable importance
				4,       // number of variables randomly selected at node and used to find the best split(s).
				100,	 // max number of trees in the forest
				0.01f,				// forrest accuracy
				CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
		);

		// train random forest classifier (using training data)

		printf( "\nUsing training database: %s\n\n", argv[1]);
		CvRTrees* rtree = new CvRTrees;

		rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
				Mat(), Mat(), var_type, Mat(), params);

		// perform classifier testing and report results

		Mat test_sample;
		int correct_class = 0;
		int wrong_class = 0;
		int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0,0,0,0,0,0,0};

		printf( "\nUsing testing database: %s\n\n", argv[2]);

		for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
		{

			// extract a row from the testing matrix

			test_sample = testing_data.row(tsample);

			// run random forest prediction

			result = rtree->predict(test_sample, Mat());

			//printf("Testing Sample %i -> class result (class %d)\n", tsample, (int) result);
			cout<<(int)result <<"\n";
			// if the prediction and the (true) testing classification are the same
			// (N.B. openCV uses a floating point decision tree implementation!)

			if (fabs(result - testing_classifications.at<float>(tsample, 0))
					>= FLT_EPSILON)
			{
				// if they differ more than floating point error => wrong class

				wrong_class++;

				false_positives[(int) result]++;

			}
			else
			{

				// otherwise correct

				correct_class++;
			}
		}

		printf( "\nResults on the testing database: %s\n"
				"\tCorrect classification: %d (%g%%)\n"
				"\tWrong classifications: %d (%g%%)\n",
				argv[2],
				correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
				wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

		for (int i = 0; i < NUMBER_OF_CLASSES; i++)
		{
			printf( "\tClass (Class %d) false postives 	%d (%g%%)\n", i,
					false_positives[i],
					(double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
		}


		// all matrix memory free by destructors


		// all OK : main returns 0

		return 0;
	}

	// not OK : main returns -1

	return -1;
}
/******************************************************************************/
