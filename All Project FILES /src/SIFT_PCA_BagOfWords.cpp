/*
Name       : Ravi Kant
USC ID     : 7945-0425-48	
e-mail     : rkant@usc.edu	
Submission : Nov 21, 2015

Input Format: programName Data_Location ClassA_name number_of_classA_samples
				classB_name number_of_classB_samples number_of_test_samples

 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <dirent.h>
#include <typeinfo>

using namespace cv;
using namespace std;

// input: input_data_file

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
int main()
{

	// parameters to set
	int num_classes = 10;
	int dimensionToReduceTo = 15;		// PCA good value 15
	int num_clusters = 300;				// Bag of words #clusters
	int k_nearest_neighbor = 10; 		// K-Nearest Neighbor
	// output
	// 		feature_train.txt
	// 		label_train.txt
	// 		feature_test.txt
	// 		sift_pca_bagOfwords_results.txt


	// Read parameters
	string raw_data_location;
	int num_training_samples_class;
	int num_testing_samples_class;

	vector<string> feature_train_image_names;
	vector<string> feature_test_image_names;

	ifstream fin;
	fin.open("dataDescription.txt");
	if(fin){
		string temp;
		getline(fin,raw_data_location);
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

	// Make a list of all valid class names
	string class_name_array[num_classes];
	fin.open("feature_class_names.txt");
	string temp;
	if(fin){

		vector<string> validClassNames;
		while(getline(fin,temp)){
			temp = temp.substr(0, temp.find("\t"));
			validClassNames.push_back(temp);
		}
		fin.close();

		if( num_classes > validClassNames.size() ){
			cout<<"\nWe do not enough classes that required number of samples. \nPlease reduce the"
					"number of training and/or test samples you want to use";
		}
		else {
			for(int i = 0; i < num_classes; i++) {
				class_name_array[i] = validClassNames[i];
			}
			fin.open("feature_train_image_names.txt");
			if(fin){
				string temp;
				while(getline(fin,temp)){
					feature_train_image_names.push_back(temp);
				}
				fin.close();

			}
			else
			{
				cout<<"Unable to open feature_train_image_names.txt \nPlease run randomDataSubSampler.cpp first\n";
				exit(1);
			}
			fin.open("feature_test_image_names.txt");
			if(fin){
				string temp;
				while(getline(fin,temp)){
					feature_test_image_names.push_back(temp);
				}
				fin.close();

			}
			else
			{
				cout<<"Unable to open feature_test_image_names.txt \nPlease run randomDataSubSampler.cpp first\n";
				exit(1);
			}
		}
	}
	else
	{
		cout<<"Unable to open feature_class_names.txt. \nPlease run randomDataSubSampler.cpp first\n";
		exit(1);
	}

	// declare space to store SIFT features in 128X(total number of keypoints)
	//vector< vector<double> > sift_feature_matrix;
	Mat sift_feature_matrix;
	// store the number of keypoints in each image
	Mat_<int> num_keypoints_matrix(num_classes,num_training_samples_class);

	// iterate over each class one by one
	int cur_class = 0;
	int cum_image_num = 0;
	for(cur_class = 0; cur_class < num_classes; cur_class++) {

		string cur_class_raw_data_location = raw_data_location + class_name_array[cur_class] + "/";

		for(int cur_image_num = 0; cur_image_num < num_training_samples_class; cum_image_num++, cur_image_num++) {

			string cur_image_location = cur_class_raw_data_location + feature_train_image_names[cum_image_num];

			Mat cur_image = imread(cur_image_location,0);

			SiftFeatureDetector detector;
			vector<cv::KeyPoint> image_keypoints;
			detector.detect(cur_image, image_keypoints);

			num_keypoints_matrix[cur_class][cur_image_num] = image_keypoints.size();

			// Calculate descriptors: For each of the key points
			// obtain the features describing the vicinity of the
			// the key points. This will be a 128 dimensional vector
			// at each key point

			SiftDescriptorExtractor extractor;
			Mat kepoint_descriptors;
			extractor.compute( cur_image, image_keypoints, kepoint_descriptors );
			sift_feature_matrix.push_back(kepoint_descriptors);

		}
	}

	// PCA to reduce dimensionality from 128 features to dimensionToReduceTo
	int reducedDimension = dimensionToReduceTo;
	PCA pca(sift_feature_matrix, Mat(), CV_PCA_DATA_AS_ROW, reducedDimension);
	Size size_sift_feature_matrix = sift_feature_matrix.size();
	Mat_<float> projected(size_sift_feature_matrix.height,reducedDimension);
	pca.project(sift_feature_matrix,projected);

	Mat_<float> pcaSIFT_feature_matrix;
	projected.convertTo(pcaSIFT_feature_matrix,CV_32F);

	// k means clustering
	// labels: vector storing the labels assigned to each vector
	//         (the pcaSIFT feature of a keypoint). Therefore labels
	//         is of size = total number of keypoints = size_sift_feature_matrix.height

	vector<int> labels;//(size_sift_feature_matrix.height);
	int attempts = 5;
	Mat centers;
	TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20000, 0.0001);

	kmeans(pcaSIFT_feature_matrix, num_clusters, labels,criteria, attempts, KMEANS_RANDOM_CENTERS,centers );

	// Object Feature Vector
	// computing histograms of each image
	// the keypoint_matrix stores the number of keypoints of each image
	// each image has a different number of keypoints
	// using this matrix, we will compute the histogram for each image
	// Also, note that the pcaSIFT_matrix stores the pcaSift_features in
	// following order:
	// 		pcaSift_feature of keypoint 1 of image 1 of class 1
	// 		pcaSift_feature of keypoint 2 of image 1 of class 1
	// 		.
	// 		.
	// 		pcaSift_feature of keypoint 1 of image 2 of class 1
	// 		pcaSift_feature of keypoint 2 of image 2 of class 1
	// 		.
	// 		.
	// 		pcaSift_feature of keypoint 1 of image 1 of class 2
	// 		.
	// 		.
	// 		.
	// 		pcaSift_feature of last keypoint of last image of last class

	Mat histogram_images = Mat(num_training_samples_class*num_classes, num_clusters, CV_32F, float(0.0));

	vector<int> labels_train(num_training_samples_class*num_classes);
	int cImg = 0;

	int min_keypoint_index = 0;
	int cumImage_index = 0;
	for(int curClass = 0; curClass < num_classes; curClass++) {
		for(int curImage = 0; curImage < num_training_samples_class; curImage++) {

			int numKeypoints = num_keypoints_matrix[curClass][curImage];

			for(unsigned int i = 0; i < numKeypoints; i++) {

				int id = labels[min_keypoint_index+i];
				histogram_images.at<float>(cumImage_index,id) += 1.0;
			}

			min_keypoint_index = min_keypoint_index + numKeypoints;
			labels_train[cumImage_index] = curClass;
			cumImage_index++;
		}

	}

	ofstream fout;
	ofstream FOUT;
	fout.open("training_feature_histogram_images.txt");
	for(int i = 0; i < num_training_samples_class*num_classes;i++){
		for(int j = 0; j < num_clusters; j++){
			fout<<histogram_images.at<float>(i, j)<<" ";
		}
		fout<<"\n";
	}
	fout.clear();
	fout.close();
	fout.open("training_label_histogram_images.txt");
	fout.clear();
	for(int i = 0; i < num_training_samples_class*num_classes; i++) {
		fout<<labels_train[i]<<"\n";
	}
	fout.close();
	// Normalize the histogram matrix
	Mat normalized_histogram_images;
	normalize(histogram_images, normalized_histogram_images);
	histogram_images = normalized_histogram_images;


	// ===============================================================
	// Read Test Images
	// ===============================================================

	Mat_<int> testing_num_keypoints_matrix(num_classes,num_testing_samples_class);
	Mat testing_sift_feature_matrix;
	int cum_image_index = 0;
	for(cur_class = 0; cur_class < num_classes; cur_class++) {

		string cur_class_raw_data_location = raw_data_location + "/" + class_name_array[cur_class] + "/";

		//read image of the testing data of the current_class one at a time

		for(int cur_image_num = 0; cur_image_num < num_testing_samples_class; cum_image_index++,cur_image_num++) {
			string cur_image_location = cur_class_raw_data_location + feature_test_image_names[cum_image_index];

			Mat cur_image = imread(cur_image_location,0);

			SiftFeatureDetector detector;
			vector<cv::KeyPoint> image_keypoints;
			detector.detect(cur_image, image_keypoints);

			testing_num_keypoints_matrix[cur_class][cur_image_num] = image_keypoints.size();

			// Calculate descriptors: For each of the key points
			// obtain the features describing the vicinity of the
			// the key points. This will be a 128 dimensional vector
			// at each key point

			SiftDescriptorExtractor extractor;
			Mat kepoint_descriptors;
			extractor.compute( cur_image, image_keypoints, kepoint_descriptors );
			testing_sift_feature_matrix.push_back(kepoint_descriptors);

		}
	}

	// Project the test image SIFT feature to the PCA reduced
	// dimension plane
	Size size_testing_sift_feature_matrix = testing_sift_feature_matrix.size();
	Mat_<float> testing_projected(size_testing_sift_feature_matrix.height,reducedDimension);
	pca.project(testing_sift_feature_matrix,testing_projected);

	Mat_<float> testing_pcaSIFT_feature_matrix;
	testing_projected.convertTo(testing_pcaSIFT_feature_matrix,CV_32F);


	Mat testing_histogram_images = Mat(num_testing_samples_class*num_classes, num_clusters, CV_32F, float(0.0));
	vector<int> labels_test(num_testing_samples_class*num_classes);
	cImg = 0;
	min_keypoint_index = 0;
	cumImage_index = 0;
	for(int curClass = 0; curClass < num_classes; curClass++) {
		for(int curImage = 0; curImage < num_testing_samples_class; curImage++) {

			int numKeypoints = testing_num_keypoints_matrix[curClass][curImage];

			Mat tempDescriptor=testing_pcaSIFT_feature_matrix(cv::Rect(0,min_keypoint_index,reducedDimension,numKeypoints));

			FlannBasedMatcher flann_matcher;
			std::vector< DMatch > flann_matches;
			flann_matcher.match( tempDescriptor, centers, flann_matches );

			for(unsigned int i = 0; i < flann_matches.size(); i++) {
				int id = flann_matches[i].trainIdx;
				testing_histogram_images.at<float>(cumImage_index,id) += 1.0;
			}

			min_keypoint_index = min_keypoint_index + numKeypoints;
			labels_test[cumImage_index] = curClass;
			cumImage_index++;
		}
	}

	fout.open("testing_feature_histogram_images.txt");
	for(int i = 0; i < num_testing_samples_class*num_classes;i++){
		for(int j = 0; j < num_clusters; j++){
			fout<<testing_histogram_images.at<float>(i, j)<<" ";
		}
		fout<<"\n";
	}
	fout.clear();
	fout.close();

	fout.open("testing_labels_histogram_images.txt");
	fout.clear();
	for(int i = 0; i < num_testing_samples_class*num_classes;i++){
		fout<<labels_test[i]<<"\n";
	}
	fout.close();
	// NORMALIZE HISTOGRAMS
	Mat normalized_testing_histogram_images;
	normalize(testing_histogram_images,normalized_testing_histogram_images);
	testing_histogram_images = normalized_testing_histogram_images;

	cout<<"\n\n===========BOW=======================\n\n";

	FlannBasedMatcher flann_matcher;
	vector< vector < DMatch > > flann_matches;

	Mat_<float> testHist = testing_histogram_images;
	Mat_<float> trainHist = histogram_images;

	flann_matcher.knnMatch( testHist, trainHist, flann_matches, k_nearest_neighbor );

	int predTestLabels[num_testing_samples_class*num_classes];
	for(int imgNo = 0; imgNo < num_testing_samples_class*num_classes; imgNo++) {
		vector < DMatch > temp = flann_matches[imgNo];

		float votes[num_clusters]={0.0};
		const int N = sizeof(votes) / sizeof(float);
		for(int neigh = 0; neigh < temp.size(); neigh++ ) {
			int id = temp[neigh].trainIdx;
			int ind = id;
			id = ind/num_training_samples_class;
			if(ind%num_training_samples_class == 0)
				id = id - 1;

			float dist = temp[neigh].distance;
			votes[id] = votes[id] + (1.0/dist);

		}
		predTestLabels[imgNo] = distance(votes, max_element(votes, votes + N));
	}


	// compute error
	vector<float> error(num_classes,0.0);
	float totalError=0.0;
	for(int i = 0; i < num_testing_samples_class*num_classes; i++) {

		if(predTestLabels[i] != labels_test[i])
		{
			error[labels_test[i]] = error[labels_test[i]] + 1.0;
			totalError = totalError + 1.0;
		}
	}


	//	ofstream fout;
	// Write to file
	fout.open("feature_test.txt");

	for(int i = 0; i < num_testing_samples_class*num_classes;i++){
		for(int j = 0; j < num_clusters; j++){
			fout<<testing_histogram_images.at<float>(i, j)<<" ";
		}
		fout<<"\n";
	}
	fout.clear();
	fout.close();


	fout.open("label_test.txt");
	fout.clear();
	for(int i = 0; i < num_testing_samples_class*num_classes;i++){
		fout<<labels_test[i]<<"\n";
	}
	fout.close();

	fout.open("feature_train.txt");
	fout.clear();
	for(int i = 0; i < num_training_samples_class*num_classes;i++){
		for(int j = 0; j < num_clusters; j++){
			fout<<histogram_images.at<float>(i, j)<<" ";
		}
		fout<<"\n";
	}
	fout.close();

	fout.open("label_train.txt");
	fout.clear();
	for(int i = 0; i < num_training_samples_class*num_classes; i++) {
		fout<<labels_train[i]<<"\n";
	}
	fout.close();

	fout.open("predictedLabels.txt");
	fout.clear();
	for(int i = 0; i < num_testing_samples_class*num_classes; i++) {
		fout<<predTestLabels[i]<<"\n";
	}
	fout.clear();
	fout<<"\nClass Wise Number of Miss-classifications ("<<num_testing_samples_class<<" test samples "
			"in each class)\n";
	cout<<"\nClass Wise Number of Miss-classifications ("<<num_testing_samples_class<<" test samples "
			"in each class)\n";
	for(int i = 0; i < num_classes; i++) {
		fout<<class_name_array[i]<<"\t: "<<error[i]<<"\n";
		cout<<class_name_array[i]<<"\t: "<<error[i]<<"\n";
	}
	fout<<"Total Error(%)\n"<<totalError*100/(num_testing_samples_class*num_classes);
	cout<<"Total Error(%) "<<totalError*100/(num_testing_samples_class*num_classes);
	fout.close();

}
