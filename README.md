# ee660
contains  the project for EE660 Machine Learning

All files are in "All Project FILES"

to run:
>>open All PRoject Files/src/
>>edit dataDescription.txt to show path of Caltech101 database, number of training and number of test images
>>Next, enter following commands in terminal
cmake .
make 

./randomDataSubSampler
./SIFT_PCA_BagOfWords
>> this runs bag of words with kNN 

>>Next run normalizer.m in Matlab
>>again in terminal enter,

./SIFT_PCA_KD_Tree
 then run KD_Tress.m
 >> this runs the KD Tree implementation
 
 then again in terminal,
 ./randomForest
>> this runs the random forest implementation

To run base line minimum distance to class mean:
run basePerformance_MinDist2ClassMean.m in MATLAB

Note: Please follow the above order

/*
Name       : Ravi Kant
USC ID     : XXXX-XXXX-XX	
e-mail     : XXXXXXXX@usc.edu	
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

using namespace cv;
using namespace std;

// input: input_data_file
int main()
{

	ifstream fin;
	fin.open("dataDescription.txt");

	string raw_data_location;
	int num_classes = 5;
	int num_training_samples = 20;
	int num_test_samples = 10;

	getline(fin, raw_data_location);

	string temp;
	getline(fin, temp);
	num_classes = atoi(temp.c_str());

	// Make a list of all the class names
	string class_name_array[num_classes];

	for(int i = 0; i < num_classes; i++) {
		getline(fin, class_name_array[i]);
	//	cout<<class_name_array[i]<<"\n";
	}
	fin.close();

	// declare space to store SIFT features in 128X(total number of keypoints)
	//vector< vector<double> > sift_feature_matrix;
	Mat sift_feature_matrix;
	// store the number of keypoints in each image
	Mat_<int> num_keypoints_matrix(num_classes,20);


	// iterate over each class one by one
	int cur_class = 0;

	for(cur_class = 0; cur_class < num_classes; cur_class++) {
	//	cout<<"read here";
		string cur_class_raw_data_location = raw_data_location + "/" + class_name_array[cur_class];
		cur_class_raw_data_location = cur_class_raw_data_location + "/" + "train/";

		string   training_image_name_array[20];
		DIR *pDIR;
		struct dirent *entry;
		int k = 0;
		if( pDIR = opendir(cur_class_raw_data_location.c_str()) ){
			while(entry = readdir(pDIR)){
				string tempName = entry->d_name;
				if( tempName.find("image")!= string::npos ){
					training_image_name_array[k] = entry->d_name;
					k++;
				}
			}
			closedir(pDIR);
		}

		//read image of the training data of the current_class one at a time
		int cur_image_num = 0;
		for(cur_image_num = 0; cur_image_num < 20; cur_image_num++) {
			string cur_image_location = cur_class_raw_data_location + training_image_name_array[cur_image_num];
		//	cout<<cur_image_location<<"\n";
			Mat cur_image = imread(cur_image_location,0);
		//	equalizeHist( cur_image, cur_image );
		//	imshow("img",cur_image);
		//	waitKey(0);

			// get the keypoints
			SiftFeatureDetector detector;
			vector<cv::KeyPoint> image_keypoints;
			detector.detect(cur_image, image_keypoints);
		//	vector<KeyPoint> imKey(image_keypoints.begin(),image_keypoints.begin()+100);
		//	image_keypoints = imKey;
			//cout<<image_keypoints.size()<<"\n";
			num_keypoints_matrix[cur_class][cur_image_num] = image_keypoints.size();

			// Calculate descriptors: For each of the key points
			// obtain the features describing the vicinity of the
			// the key points. This will be a 128 dimensional vector
			// at each key point

			SiftDescriptorExtractor extractor;
			Mat kepoint_descriptors;
			extractor.compute( cur_image, image_keypoints, kepoint_descriptors );
			sift_feature_matrix.push_back(kepoint_descriptors);

		//	Mat img;
		//	vector<KeyPoint> imKey(image_keypoints.begin(),image_keypoints.begin()+100);
		//	drawKeypoints(cur_image,imKey,img);
		//	imshow("sift_keypoints",img);
		//	waitKey(0);
		//	Size sz = kepoint_descriptors.size();
		//	cout<<sz.height<<" "<<sz.width<<"\n";
		//	exit(1);

		}

	}

	// PCA to reduce dimensionality from 128 features to 20
	int reducedDimension = 120;
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
	int num_clusters = 1200;
	vector<int> labels;//(size_sift_feature_matrix.height);
	int attempts = 5;
	Mat centers;
	TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);

	kmeans(pcaSIFT_feature_matrix, num_clusters, labels,criteria, attempts, KMEANS_RANDOM_CENTERS,centers );
//	cout<<labels.size()<<"===========";
//	exit(0);
//	for(int p= 0; p<200;p++)
//		cout<<labels[p]<<"\n";

	// Object Feature Vector
	// computing histograms of each image
	// the keypoint_matrix stores the number of keypoints of each image
	// each image has a different number of keypoints
	// using this matrix, we will compute the histogram for each image
	// Also, note that the pcaSIFT_matrix stores the pcaSift_features in
	// following order:
	// pcaSift_feature of keypoint 1 of image 1 of class 1
	// pcaSift_feature of keypoint 2 of image 1 of class 1
	// .
	// .
	// pcaSift_feature of keypoint 1 of image 2 of class 1
	// pcaSift_feature of keypoint 2 of image 2 of class 1
	// .
	// .
	// pcaSift_feature of keypoint 1 of image 1 of class 2
	// .
	// .
	// .
	// pcaSift_feature of last keypoint of last image of last class

	Mat histogram_images = Mat(20*num_classes, num_clusters, CV_64F, double(0));
	vector<int> labels_train(20*num_classes);
	int cImg = 0;

	int min_keypoint_index = 0;
	int cumImage_index = 0;
	for(int curClass = 0; curClass < num_classes; curClass++) {
			for(int curImage = 0; curImage < 20; curImage++) {
		//		if(cumImage_index==100)
		//						break;
			//	cout<<num_keypoints_matrix[curClass][curImage]<<" ";
				int numKeypoints = num_keypoints_matrix[curClass][curImage];

				//	cout<<"start:"<<min_keypoint_index<<"\tfinish:"<<max_keypoint_index<<"\n";

				Mat tempDescriptor=pcaSIFT_feature_matrix(cv::Rect(0,min_keypoint_index,reducedDimension,numKeypoints));

				FlannBasedMatcher flann_matcher;
				std::vector< DMatch > flann_matches;
				flann_matcher.match( tempDescriptor, centers, flann_matches );
			//	if(curClass == 0 && curImage == 0)
			//		cout<<flann_matches.size();
				 for(unsigned int i = 0; i < flann_matches.size(); i++) {
					 int id = flann_matches[i].trainIdx;
					histogram_images.at<double>(cumImage_index,id) += 1;
				}
			/*	for(int keypoint_index = min_keypoint_index; keypoint_index < max_keypoint_index; keypoint_index++ ){
					int keypoint_label = labels[keypoint_index];
					histogram_images.at<double>(cumImage_index,keypoint_label) += 1;
				}*/
				min_keypoint_index = min_keypoint_index + numKeypoints;
				labels_train[cumImage_index] = curClass;
				cumImage_index++;
			}
	}
//	cout<<histogram_images.row(0);
//exit(0);
//	cout<<"\nWe are good";
	// histogram_images now contains the new representation of each image
/*	ofstream fout;
	fout.open("feature_train.txt");

	for(int i = 0; i < 20*num_classes;i++){
		for(int j = 0; j < num_clusters; j++){
			fout<<histogram_images.at<double>(i, j)<<",";
		}
		fout<<"\n";
	}
	fout.close();

	fout.open("label_train.txt");
	for(int i = 0; i < 20*num_classes; i++) {
		fout<<labels_train[i]<<"\n";
	}
	fout.close();*/

	// ===============================================================
	// Read Test Images
	// ===============================================================
	//cout<<"DDDDDDDDD\n";



	Mat_<int> testing_num_keypoints_matrix(num_classes,10);
	Mat testing_sift_feature_matrix;
	for(cur_class = 0; cur_class < num_classes; cur_class++) {

		string cur_class_raw_data_location = raw_data_location + "/" + class_name_array[cur_class];
		cur_class_raw_data_location = cur_class_raw_data_location + "/" + "test/";

		string   testing_image_name_array[10];
		DIR *pDIR;
		struct dirent *entry;
		int k = 0;
		if( pDIR = opendir(cur_class_raw_data_location.c_str()) ){
			while(entry = readdir(pDIR)){
				string tempName = entry->d_name;
				if( tempName.find("image")!= string::npos ){
					testing_image_name_array[k] = entry->d_name;
					k++;
				}
			}
			closedir(pDIR);
		}
	//	cout<<"DONE\n";


		//read image of the testing data of the current_class one at a time
		int cur_image_num = 0;
		for(cur_image_num = 0; cur_image_num < 10; cur_image_num++) {
			string cur_image_location = cur_class_raw_data_location + testing_image_name_array[cur_image_num];
			//	cout<<cur_image_location<<"\n";
			Mat cur_image = imread(cur_image_location,0);

			//	imshow("img",cur_image);
			// waitKey(0);

			// get the keypoints
			SiftFeatureDetector detector;
			vector<cv::KeyPoint> image_keypoints;
			detector.detect(cur_image, image_keypoints);
		//	vector<KeyPoint> imKey(image_keypoints.begin(),image_keypoints.begin()+100);
			//image_keypoints = imKey;
			//cout<<image_keypoints.size()<<"\n";
			testing_num_keypoints_matrix[cur_class][cur_image_num] = image_keypoints.size();

			// Calculate descriptors: For each of the key points
			// obtain the features describing the vicinity of the
			// the key points. This will be a 128 dimensional vector
			// at each key point

			SiftDescriptorExtractor extractor;
			Mat kepoint_descriptors;
			extractor.compute( cur_image, image_keypoints, kepoint_descriptors );
			testing_sift_feature_matrix.push_back(kepoint_descriptors);
			//	Size sz = kepoint_descriptors.size();
			//	cout<<sz.height<<" "<<sz.width<<"\n";
			//	exit(1);

		}
	}
	//cout<<"DONEEEEEEEEEEEEEEEEEEEEEEEEEE";


	// Project the test image SIFT feature to the PCA reduced
	// dimension plane
	Size size_testing_sift_feature_matrix = testing_sift_feature_matrix.size();
	Mat_<float> testing_projected(size_testing_sift_feature_matrix.height,reducedDimension);
	pca.project(testing_sift_feature_matrix,testing_projected);

	Mat_<float> testing_pcaSIFT_feature_matrix;
	testing_projected.convertTo(testing_pcaSIFT_feature_matrix,CV_32F);


	Mat testing_histogram_images = Mat(10*num_classes, num_clusters, CV_64F, double(0));
	vector<int> labels_test(10*num_classes);
	cImg = 0;

	min_keypoint_index = 0;
	cumImage_index = 0;
	for(int curClass = 0; curClass < num_classes; curClass++) {
			for(int curImage = 0; curImage < 10; curImage++) {
		//		if(cumImage_index==100)
		//						break;
			//	cout<<num_keypoints_matrix[curClass][curImage]<<" ";
				int numKeypoints = testing_num_keypoints_matrix[curClass][curImage];

				//	cout<<"start:"<<min_keypoint_index<<"\tfinish:"<<max_keypoint_index<<"\n";

				Mat tempDescriptor=testing_pcaSIFT_feature_matrix(cv::Rect(0,min_keypoint_index,reducedDimension,numKeypoints));

				FlannBasedMatcher flann_matcher;
				std::vector< DMatch > flann_matches;
				flann_matcher.match( tempDescriptor, centers, flann_matches );
			//	if(curClass == 0 && curImage == 0)
			//		cout<<flann_matches.size();
				 for(unsigned int i = 0; i < flann_matches.size(); i++) {
					 int id = flann_matches[i].trainIdx;
					 testing_histogram_images.at<double>(cumImage_index,id) += 1;
				}
			/*	for(int keypoint_index = min_keypoint_index; keypoint_index < max_keypoint_index; keypoint_index++ ){
					int keypoint_label = labels[keypoint_index];
					histogram_images.at<double>(cumImage_index,keypoint_label) += 1;
				}*/
				min_keypoint_index = min_keypoint_index + numKeypoints;
				labels_test[cumImage_index] = curClass;
				cumImage_index++;
			}
	}


	cout<<"\n\n===========TTTTTT=======================\n\n";
	FlannBasedMatcher flann_matcher;
	vector< vector < DMatch > > flann_matches;
/*	Mat test_histogram(10*num_classes,num_clusters,CV_32F,testing_histogram_images);

	Mat histImg(20*num_classes,num_clusters,CV_32F)=histogram_images;*/
	Mat_<float> testHist = testing_histogram_images;
	Mat_<float> trainHist = histogram_images;
	flann_matcher.knnMatch( testHist, trainHist, flann_matches,10 );
	//cout<<"\nGGGGGGGGG\n";
	int testLabels[10*num_classes];
	for(int imgNo = 0; imgNo < 10*num_classes; imgNo++) {
		vector < DMatch > temp = flann_matches[imgNo];
		//cout<<"#";
		float votes[num_clusters]={0};
		const int N = sizeof(votes) / sizeof(float);
		for(int neigh = 0; neigh < temp.size(); neigh++ ) {
			int id = temp[neigh].trainIdx;
			if(id<20)
				id = 0;
			else if(id<40)
				id = 1;
			else if(id < 60)
				id = 2;
			else if(id<80)
				id = 3;
			else id = 4;
			float dist = temp[neigh].distance;
			votes[id] = votes[id] + (1.0/dist);
		//	cout<<"=";
		}
		testLabels[imgNo] = distance(votes, max_element(votes, votes + N));
		cout<<testLabels[imgNo]<<" ";
	}



	/*fout.open("testing_pcaSIFT_feature_matrix.txt");
	for(int i = 0; i < size_testing_sift_feature_matrix.height; i++) {
		for(int j = 0; j < reducedDimension; j++) {
			fout<<testing_pcaSIFT_feature_matrix.at<double>(i,j)<<",";
		}
		fout<<"\n";
	}
	fout.close();

	fout.open("cluster_centers.txt");
	Size size_centers = centers.size();
	for(int i = 0; i < size_centers.height; i++) {
		for(int j = 0; j <size_centers.width; j++) {
			fout<<centers.at<double>(i,j)<<",";
		}
		fout<<"\n";
	}
	fout.close();
	fout.open("testing_num_keypoints_matrix.txt");
	for(int i = 0; i < num_classes; i++) {
		for(int j = 0; j < 10; j++) {
			fout<<testing_num_keypoints_matrix.at<int>(i,j)<<" ";
		}
		fout<<"\n";
	}
	fout.close();*/

}// end of main
