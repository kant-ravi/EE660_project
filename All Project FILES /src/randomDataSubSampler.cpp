/*
Name       : Ravi Kant
USC ID     : 7945-0425-48	
e-mail     : rkant@usc.edu	
Submission : Nov 21, 2015

Input Format: programName Data_Location ClassA_name number_of_classA_samples
				classB_name number_of_classB_samples number_of_test_samples

 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <dirent.h>
#include <typeinfo>
#include <ctime>
#include <cstdlib>
#include <algorithm>
using namespace std;

// input  : None. reads dataDescription.txt
// output : feature_class_names					// names of classes that have sufficient samples
//			feature_train_image_names			// num_training_samples_class number of names of images from each class
//			feature_test_image_names			// num_testing_samples_class number of names of images from each class
int main()
{
	// read dataDescription.txt
	string raw_data_location;
	int num_training_samples_class;
	int num_testing_samples_class;

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
	//Output
	vector<string> feature_class_names;
	vector<string> feature_train_image_names;
	vector<string> feature_test_image_names;

	// Read folder, determine classNames
	vector<string> class_name_array;
	DIR *pDIR;
	struct dirent *entry;
	int k = 0;
	if( pDIR = opendir(raw_data_location.c_str()) ){

		while(entry = readdir(pDIR)){
			string tempName = entry->d_name;
			if( !(tempName.find(".")!= string::npos) ){
				class_name_array.push_back( entry->d_name );
				k++;
			}
		}
		closedir(pDIR);
	}
	srand(time(NULL));

	// select training and testing images from each class
	vector<int> numSamplesClass;
	for(int cur_class = 0; cur_class < class_name_array.size(); cur_class++) {
		string cur_class_name = class_name_array[cur_class];
		string class_data_location = raw_data_location + cur_class_name + "/";

		vector<string> class_image_name_array;
		k = 0;
		if( pDIR = opendir(class_data_location.c_str()) ){
			while(entry = readdir(pDIR)){
				string tempName = entry->d_name;
				if( (tempName.find("image")!= string::npos) ){
					class_image_name_array.push_back(entry->d_name);
					k++;
				}
			}
			closedir(pDIR);

			int num_images_cur_class = class_image_name_array.size();

			// if we have sufficient examples separate randomly into test and training

			if(num_images_cur_class >= num_training_samples_class+num_testing_samples_class) {

				feature_class_names.push_back( cur_class_name );
				numSamplesClass.push_back(num_images_cur_class);
				int max = num_images_cur_class;
				int min = 0;

				int indices[max];
				for(int i = 0; i < max; i++)
					indices[i] = i;
				random_shuffle(&(indices[0]), &(indices[max]));
				int chkList[max]={0};
				int total_training_samples = 0, total_test_sample = 0;
				for(int i = 0; i < max;i++)
				{
					if(total_training_samples < num_training_samples_class){
						total_training_samples++;
						feature_train_image_names.push_back(class_image_name_array[indices[i]]);
						continue;
					}

					if(total_test_sample < num_testing_samples_class) {
						total_test_sample++;
						feature_test_image_names.push_back(class_image_name_array[indices[i]]);
						continue;
					}
				}
			}
		}
	}

	ofstream fout;
	fout.open("feature_class_names.txt");

	for(int i = 0; i < feature_class_names.size();i++){
		fout<<feature_class_names[i]<<"\t"<<numSamplesClass[i]<<"\n";
	}
	fout.close();

	fout.open("feature_train_image_names.txt");
	for(int i = 0; i < feature_train_image_names.size();i++){
		fout<<feature_train_image_names[i]<<"\n";
	}
	fout.close();
	fout.open("feature_test_image_names.txt");
	for(int i = 0; i < feature_test_image_names.size();i++){
		fout<<feature_test_image_names[i]<<"\n";
	}
	fout.close();

}
