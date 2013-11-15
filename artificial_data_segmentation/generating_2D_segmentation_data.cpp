#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using namespace cv;

int get(int* ar,int s,int x,int y){
	if(x<0 || x>=s || y<0 || y>=s)
		std::cout << "error in get()" << std::endl;
	return ar[x*s+y];
}

void set(int* ar,int s,int x,int y,int v){
	if(x<0 || x>=s || y<0 || y>=s)
		std::cout << "error in set()" << std::endl;
	ar[x*s+y]=v;
}

static Scalar randomColor( RNG& rng ){
	int icolor = (unsigned) rng;
	return Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
}

int main( int argc, char** argv )
{
	/*char* imageName = argv[1];
	Mat image;
	if( argc != 2)
	{
		printf( " No image data \n " );
		return -1;
	}
	image = imread( imageName, 1 );
	if( !image.data )
	{
		printf( " No image data \n " );
		return -1;
	}*/
	if( argc != 2)
	{
		printf( " requires one argument, the experiment identifier (not repeatable!) \n" );
		return -1;
	}
	srand (time(NULL));
	int imsize=40;
	/*Mat segments(imsize,imsize, CV_8UC3, Scalar(0,0,0));
	namedWindow( "image", CV_WINDOW_AUTOSIZE );
	imshow( "image", segments );
	waitKey(0);*/

	int segments[imsize*imsize];
	for(int i=0;i<imsize*imsize;i++)
		segments[i]=0;

	int nsegments=rand() % 30 + 30;//between 30 and 59
	std::cout << "generating " << nsegments << " random segments" << std::endl;

	//place each in a random pixel which isn't already occupied
	for (int i=0;i<nsegments;){
		int x=rand() % imsize;
		int y=rand() % imsize;
		if(get(segments,imsize,x,y)==0){
			set(segments,imsize,x,y,i);
			i++;
			//std::cout << "p[0] set to " << get(segments,imsize,x,y) << std::endl;
		}
		//else
			//std::cout << "p[0] already set to " << get(segments,imsize,x,y) << std::endl;
	}
	
	/*for(int i=0;i<imsize;i++){
		for(int j=0;j<imsize;j++)
			std::cout << get(segments,imsize,i,j) << "\t";
		std::cout << std::endl;
	}*/

	
	//itterate for a long time, and expand each segment into adjacent 0s
	for (int n=0;n<imsize*imsize-nsegments+1;){
		//find a random pixel which is set to 0
		int x=rand() % imsize;
		int y=rand() % imsize;
		while(get(segments,imsize,x,y)!=0){
			x=rand() % imsize;
			y=rand() % imsize;
		}
		//look at surrounding pixels, and assign this pixel the value of the first non-zero
		//std::cout << "at " << x << ", " << y << std::endl;
		for (int i=max(0,x-1);i<min(x+2,imsize);i++)
			for (int j=max(0,y-1);j<min(y+2,imsize);j++){
				//only look for adjacent, not diagonal
				//if(x==i || y==j){
					//std::cout << "check " << i << ", " << j << std::endl;
					if(get(segments,imsize,i,j)!=0){
						//std::cout << "change for " << get(segments,imsize,i,j) << std::endl;
						set(segments,imsize,x,y,get(segments,imsize,i,j));
						//better than break:
						j=imsize;
						i=imsize;
						n++;

						//std::cout << "segments = " << std::endl << " " << segments << std::endl << std::endl;
					}
				//}
			}
	}
	/*std::cout << "nsegments: " << nsegments << std::endl;
	for(int i=0;i<imsize;i++){
		for(int j=0;j<imsize;j++)
			std::cout << get(segments,imsize,i,j) << "\t";
		std::cout << std::endl;
	}*/

	//generate #nsegments random colors
	RNG rng( rand() );
	Scalar colors[nsegments];
	for (int i=0;i<nsegments;i++)
		colors[i]=randomColor(rng);

	Mat segments_im(imsize,imsize, CV_8UC3, Scalar(0,0,0));

	for(int i=0;i<imsize;i++){
		for(int j=0;j<imsize;j++){
			Mat roi(segments_im, Rect(i,j,1,1));
			roi = colors[get(segments,imsize,i,j)];
		}
	}
//nsegments=1;
	//for each segment, define a Gaussian distribution at random
	double means[nsegments];
	double variances[nsegments];
	for (int i=0;i<nsegments;i++){
		means[i]=((((double)rand())/RAND_MAX))*255;
		variances[i]=(((double)rand())/RAND_MAX)*50;
		std::cout << "segment #" << i << " has mu: " << means[i] << ", sigma: " << variances[i] << std::endl;
	}

	std::default_random_engine generator;
	for(int n=0;n<10;n++){
		//over #samples, sample from the distribution of each segment
		double sample[nsegments];
		for (int i=0;i<nsegments;i++){
			std::normal_distribution<double> distribution(means[i],variances[i]);
			sample[i] = distribution(generator);

			std::cout << "segment " << i << ": " << sample[i] << std::endl;
		}

		Mat M(imsize, imsize, 0);

		//add noise to each pixel
		std::normal_distribution<double> distribution(0,30);
		for (int i=0;i<imsize;i++){
			for (int j=0;j<imsize;j++){
				Mat roi(M, Rect(i,j,1,1));
				int seg = get(segments,imsize,i,j);
				int val = round(sample[seg]+distribution(generator));
				//I wish I knew why max(0,min(255,round(#)) doesn't compile;
				if(val<0)
					val=0;
				if(val>255)
					val=255;
				roi = val;
			}
		}
		/*namedWindow( "image", CV_WINDOW_AUTOSIZE );
		imshow( "image", M );

		waitKey(0);*/
		
		//output each of #samples images
		std::stringstream ss;
		ss << "generated_data/segmented_2D_" << argv[1] << "_" << (1+n) << ".jpg";
		imwrite(ss.str(), M);
	}

	return 0;
}
