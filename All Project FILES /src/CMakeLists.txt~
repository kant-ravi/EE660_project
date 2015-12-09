cmake_minimum_required(VERSION 2.8)
project( USC_Sem3_ML_project )
find_package( OpenCV REQUIRED )

add_executable( randomDataSubSampler randomDataSubSampler.cpp )
target_link_libraries( randomDataSubSampler ${OpenCV_LIBS} )

add_executable( SIFT_PCA_BagOfWords SIFT_PCA_BagOfWords.cpp )
target_link_libraries( SIFT_PCA_BagOfWords ${OpenCV_LIBS} )

add_executable( randomForest randomForest.cpp )
target_link_libraries( randomForest ${OpenCV_LIBS} )

add_executable( SIFT_PCA_NaiveBayes SIFT_PCA_NaiveBayes.cpp )
target_link_libraries( SIFT_PCA_NaiveBayes ${OpenCV_LIBS} )

